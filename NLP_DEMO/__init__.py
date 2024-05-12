from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)



for param in model.base_model.parameters():
    param.requires_grad = False