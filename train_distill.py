import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

class RationaleDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the input query
        input_encoding = self.tokenizer(
            item['query'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Tokenize the target label and rationale combined
        target_text = f"Label: {item['label']} Reason: {item['rationale']}"
        target_encoding = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Flatten tensors
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()
        
        # Replace padding token id's of the labels by -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class MultiTaskRouterModel(torch.nn.Module):
    def __init__(self, model_name="t5-base"):
        super(MultiTaskRouterModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    def generate(self, input_text):
        self.model.eval()
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        return generated_text

def train_model():
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    dataset = RationaleDataset('rationale_dataset.json', tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MultiTaskRouterModel()
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    for epoch in range(3):  # Number of epochs
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/3, Loss: {avg_loss}")
    
    model.model.save_pretrained('/opt/ml/model')  # Save the model after training
    tokenizer.save_pretrained('/opt/ml/model')     # Save tokenizer as well


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model()
