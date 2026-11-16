"""Train GPT-2 on cleaned course groups using config.yaml.

Run from the repository root: python train_cpt.py
Place cleaned page_*.txt files in data/cpt/ first.
"""
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    with Path('config.yaml').open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    settings = config['cpt']
    seed = int(settings['seed'])
    length = int(settings['sequence_length'])
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = Path('data/cpt')
    pages = sorted(folder.glob('page_*.txt'))
    held_out = settings['validation_pages']
    if not pages:
        raise FileNotFoundError('Add cleaned page_*.txt files to data/cpt/.')
    for name in held_out:
        if not (folder / name).is_file():
            raise FileNotFoundError(f'Missing validation page: {name}')
    train_pages = [page for page in pages if page.name not in held_out]
    if not train_pages:
        raise ValueError('No training pages remain after the split.')

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    text = ''.join(page.read_text(encoding='utf-8') + '\n' for page in train_pages)
    import re
    groups = []
    for row in re.split(r'(?m)(?=^\d{5,}\s+[A-Za-z])', text):
        if not row.strip():
            continue
        if groups and re.search(
            r'\b\d+\s+(Prerequisite|Equivalent|Concurrent|Pre\./Con\.)', row
        ):
            groups[-1] += '\n' + row
        else:
            groups.append(row)
    blocks = [tokenizer.encode(group) + [tokenizer.eos_token_id] for group in groups]
    if any(len(block) > length for block in blocks):
        raise ValueError('A course group exceeds the configured sequence length.')
    ids = torch.full((len(blocks), length), tokenizer.eos_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, block in enumerate(blocks):
        ids[index, :len(block)] = torch.tensor(block)
        mask[index, :len(block)] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    loader = DataLoader(list(zip(ids, mask, labels)),
                        batch_size=int(settings['batch_size']), shuffle=True)

    results = Path('results')
    results.mkdir(parents=True, exist_ok=True)
    (results / 'split.json').write_text(json.dumps({
        'train': [page.name for page in train_pages], 'validation': held_out
    }, indent=2), encoding='utf-8')
    (results / 'cpt_data.json').write_text(json.dumps({
        'train_blocks': blocks, 'validation_pages': held_out
    }, indent=2), encoding='utf-8')

    # Reset before loading fresh GPT-2, as in the original training cell.
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(config['model_name']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings['learning_rate']))
    validation = []
    for name in held_out:
        item = tokenizer((folder / name).read_text(encoding='utf-8'),
                         return_tensors='pt').to(device)
        if item['input_ids'].shape[1] < 2:
            raise ValueError(f'Validation page has fewer than two tokens: {name}')
        if item['input_ids'].shape[1] > model.config.n_positions:
            raise ValueError('Validation page exceeds the model context window.')
        validation.append(item)

    def evaluate():
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for item in validation:
                count = item['input_ids'].shape[1] - 1
                loss = model(**item, labels=item['input_ids']).loss.item()
                total_loss += loss * count
                total_tokens += count
        loss = total_loss / total_tokens
        return loss, math.exp(loss)

    base_loss, base_ppl = evaluate()
    (results / 'base_metrics.json').write_text(json.dumps({
        'model': config['model_name'], 'loss': base_loss, 'perplexity': base_ppl,
        'validation_pages': held_out
    }, indent=2), encoding='utf-8')
    print(f'Device: {device}; groups: {len(groups)}; batches: {len(loader)}')
    print(f'Base validation perplexity: {base_ppl:.2f}')
    history = []
    best_loss, best_epoch = float('inf'), 0
    for epoch in range(1, int(settings['max_epochs']) + 1):
        model.train()
        losses = []
        for batch_ids, batch_mask, batch_labels in loader:
            optimizer.zero_grad()
            loss = model(input_ids=batch_ids.to(device),
                         attention_mask=batch_mask.to(device),
                         labels=batch_labels.to(device)).loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_loss, ppl = evaluate()
        history.append({'epoch': epoch, 'training_losses': losses,
                        'validation_loss': val_loss, 'perplexity': ppl})
        checkpoint = Path(f'cpt_epoch_{epoch}')
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        (results / 'cpt_history.json').write_text(json.dumps(history, indent=2),
                                                encoding='utf-8')
        print(f'Epoch {epoch}: train={sum(losses)/len(losses):.4f}, '
              f'validation={val_loss:.4f}, perplexity={ppl:.2f}')
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_path = Path(settings['output_dir'])
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
        elif epoch >= int(settings['early_stopping_min_epoch']):
            print('Stopped: validation loss stopped improving.')
            break
    print(f'Best epoch: {best_epoch}; best perplexity: {math.exp(best_loss):.2f}')


if __name__ == '__main__':
    main()
