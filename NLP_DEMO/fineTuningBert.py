'''
参加天池的一个学习赛，学习了利用Hugging face上的Bert模型进行微调
主要的逻辑:
BertModel和Bertforsequenceclassification都是基于Transformer架构的预训练模型，但是它们在预测任务上的用途不同。
BertModel用于生成BERT的主要表征，
而Bertforsequenceclassification则是在BERT的顶部添加一个全连接分类器用于分类任务。
之前，我不知道都是用BertModel用于生成BERT的主要表征，然后自己加一层MLP效果不怎么样最多0.78的准确率
这个代码是整理下一些类，让我以后也可以用
'''
import dataclasses
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import torch
from datasets import load_dataset
import random

from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, DefaultDataCollator, InputFeatures, \
    PreTrainedModel, AdamW, get_linear_schedule_with_warmup
import json

from transformers.trainer_pt_utils import get_parameter_names


# from NLP_DEMO.BaseLine_BERT import create_optimizer_and_scheduler, simple_accuracy, DataTrainingArguments
@dataclass
class DataTrainingArguments:
    model_dir: str = field(
        default='bert-base-chinese',
        metadata={'help': 'The pretrained model directory'}
    )
    data_dir: str = field(
        default='G:\\数据挖掘\\Pycharm_Code\\Boston_house_price\\NLP_DEMO\\download',
        metadata={'help': 'The data directory'}
    )
    max_length: int = field(
        default=64,
        metadata={'help': 'Maximum sequence length allowed to input'}
    )

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='G:\\数据挖掘\\Pycharm_Code\\Boston_house_price\\NLP_DEMO\\output\\',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'}
    )
    train_batch_size: int = field(
        default=16,
        metadata={'help': 'batch size for training'}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={'help': 'batch size for evaluation'}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Number of updates steps to accumulate before performing a backward/update pass.'}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "The total number of training epochs"}
    )
    learning_rate: float = field(
        default=3e-5,
        metadata={'help': '"The initial learning rate for AdamW.'}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    dataloader_num_workers: int = field(
        default=2,
        metadata={"help": "Number of subprocesses to use for data loading (PyTorch only)"}
    )

    logging_steps: int = field(
        default=100,
        metadata={'help': 'logging states every X updates steps.'}
    )
    eval_steps: int = field(
        default=250,
        metadata={'help': 'Run an evaluation every X steps.'}
    )
    device: str = field(
        default='cpu',
        metadata={"help": 'The device used for training'}
    )

    def get_warmup_steps(self, num_training_steps):
        return int(num_training_steps * self.warmup_ratio)

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


def create_optimizer_and_scheduler(
    args: TrainingArguments,
    model: PreTrainedModel,
    num_training_steps: int,
):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=args.get_warmup_steps(num_training_steps)
    )

    return optimizer, scheduler


access_token = "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"
#加载字典和分词工具
# token = BertTokenizer.from_pretrained('bert-base-chinese')

@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='output_data/',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'}
    )
    train_batch_size: int = field(
        default=16,
        metadata={'help': 'batch size for training'}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={'help': 'batch size for evaluation'}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Number of updates steps to accumulate before performing a backward/update pass.'}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "The total number of training epochs"}
    )
    learning_rate: float = field(
        default=3e-5,
        metadata={'help': '"The initial learning rate for AdamW.'}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading (PyTorch only)"}
    )

    logging_steps: int = field(
        default=100,
        metadata={'help': 'logging states every X updates steps.'}
    )
    eval_steps: int = field(
        default=250,
        metadata={'help': 'Run an evaluation every X steps.'}
    )
    device: str = field(
        default='cpu',
        metadata={"help": 'The device used for training'}
    )

    def get_warmup_steps(self, num_training_steps):
        return int(num_training_steps * self.warmup_ratio)

    def __str__(self):
        self_as_dict = dataclasses.asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"



# 对于NLP，之前我用的方法来设置DataSet感觉很傻，现在直接返回feature比较好不用再自定义collate_fn，不过这里的featrue其实是个类，这个是python的写法就是java 返回对象
class Dataset(torch.utils.data.Dataset):
    def __init__(self, record_id, query1, query2, labels, tokenizer,max_length=40):
        #         super(self)
        self.record_id = record_id
        self.query1 = query1
        self.query2 = query2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.record_id)

    '''
    返回这条记录的id，还有encode好的feature其实就是 embeding好的向量，attendtion score，mask
    Python这里，调用方法的时候可以输入类，然后直接实例化类，我想这就是这里这么写的原因
    '''
    def __getitem__(self, i):
        inputs = self.tokenizer(
            text=self.query1[i],
            text_pair=self.query2[i],# 这里搞错了导致我出来的embedding的shape 编程 batch size+ 15000+64 因为我每次输入dataset的都是全局数据
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        # print('row 255 input is :',inputs)

        feature = InputFeatures(**inputs, label=self.labels[i])
        # print('row 258 feature is :', feature)

        return feature


def data(env):
    # 读json文件（以训练集为例）：
    path = 'G:/数据挖掘/Pycharm_Code/Boston_house_price/NLP_DEMO/download/KUAKE-QQR_train.json'

    if env =='validation' :
        path = 'G:/数据挖掘/Pycharm_Code/Boston_house_price/NLP_DEMO/download/KUAKE-QQR_dev.json'
    else:
        if env =='test' :
            path = 'G:/数据挖掘/Pycharm_Code/Boston_house_price/NLP_DEMO/downloadKUAKE-QQR_test.json'
        else:
            path = 'G:/数据挖掘/Pycharm_Code/Boston_house_price/NLP_DEMO/download/KUAKE-QQR_train.json'


    with open(path, encoding='utf-8') as input_data:
        json_content = json.load(input_data)
        # 逐条读取记录
        record_id = []
        query1 = []
        query2 = []
        labels = []

        for block in json_content:
            if block['label'] != 'NA':
                record_id.append(block['id'])
                query1.append(block['query1'])
                query2.append(block['query2'])
                labels.append(int(block['label']))
        return record_id, query1, query2, labels

"""
这是唯一一个我看不懂的函数了，写的太简单了后面可以debug下不懂就算了
"""
def _prepare_input(data: Union[torch.Tensor, Any], device: str = 'cpu'):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data

def evaluate(
        args: TrainingArguments,
        model: PreTrainedModel,#有点意思哈，这里就是java的传入类对象..
        eval_dataloader
):
    model.eval()
    loss_list = []
    preds_list = []
    labels_list = []

    for item in eval_dataloader:
        inputs = _prepare_input(item, device=args.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss
            loss_list.append(loss.detach().cpu().item())

            preds = torch.argmax(outputs.logits.cpu(), dim=-1).numpy()
            preds_list.append(preds)

            labels_list.append(inputs['labels'].cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    loss = np.mean(loss_list)
    accuracy = simple_accuracy(preds, labels)

    model.train()

    return loss, accuracy



def train(
    args:TrainingArguments,
    model,
    tokenizer,
    train_dataset,
    dev_dataset,
    data_collator,
):
    data_collator = DefaultDataCollator()
    # initialize dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator
    )
    num_examples = len(train_dataloader.dataset)
    total_train_batch_size = args.gradient_accumulation_steps * args.train_batch_size
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    num_train_epochs = math.ceil(args.num_train_epochs)
    num_train_samples = len(train_dataset) * args.num_train_epochs

    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        args, model, num_training_steps=max_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {num_examples}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_steps}")

    model.zero_grad()
    model.train()
    t_loss = 0.0
    global_steps = 0

    best_metric = 0.0
    best_steps = -1

    for epoch in range(num_train_epochs):

        for step, item in enumerate(train_dataloader): #enumerate 默认第一个拿出的是id就是当前第几个那样子，第二个参数，就是dataset getItem我自己写的
            inputs = _prepare_input(item, device=args.device)
            # print(type(inputs))
            # for key in inputs:
            #     print(key )
            #
            # labels = key[0]
            # input_ids = key[1]
            # attention_mask = key[2]
            # token_type_ids = key[3]
            # print(type(labels))
            # print(type(input_ids))
            # print(type(attention_mask))
            # print(type(token_type_ids))
            # input_shape = input_ids.size()
            # print('input_ids.size()=',input_shape)
            # print('input_shape type =', type(input_shape))
            # input_ids is not None
            # input_shape
            # torch.Size([16, 15000, 64]) ---这个的15000是多余的，所以导致我得问题



            outputs = model(**inputs, return_dict=True)
            # outputs = model(labels=labels,input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=True)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 0:
                loss /= args.gradient_accumulation_steps

            loss.backward()
            t_loss += loss.detach()

            if (step + 1) % args.gradient_accumulation_steps == 0: #这里想做的是:目前16个样本一个batch训练，正常应该是每个batch 训练然后得到loss然后fallback，为什么这里可以累积多个batch的loss然后再计算梯度呢?
                optimizer.step()
                lr_scheduler.step()#这是干嘛用的，最顶层?#这叫学习率调度器，可以变化学习率，目前是每个batch 变化一次，也可以每个epoch变化一次
                                   #此外我还可以看到当前 学习率optimizer.param_groups[0]['lr']

                model.zero_grad()
                global_steps += 1

                if global_steps % args.logging_steps == 0:
                    print(
                        f'Training: Epoch {epoch + 1}/{num_train_epochs} - Step {(step + 1) // args.gradient_accumulation_steps} - Loss {t_loss}')

                t_loss = 0.0

            if (global_steps + 1) % args.eval_steps == 0:

                loss, acc = evaluate(args, model, dev_dataloader)
                print(
                    f'Evaluation: Epoch {epoch + 1}/{num_train_epochs} - Step {(global_steps + 1) // args.gradient_accumulation_steps} - Loss {loss} - Accuracy {acc}')

                if acc > best_metric:
                    best_metric = acc
                    best_steps = global_steps

                    saved_dir = os.path.join(args.output_dir, f'checkpoint-{best_steps}')
                    os.makedirs(saved_dir, exist_ok=True)
                    model.save_pretrained(saved_dir)
                    tokenizer.save_vocabulary(save_directory=saved_dir)

    return best_steps, best_metric




def main():
    module_name = 'hfl/chinese-bert-wwm-ext'

    module_name_base = 'bert-base-chinese'

    access_token = "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"

    device = 'cpu'

    data_args = DataTrainingArguments()
    training_args = TrainingArguments()

    # 加载序列化器
    tokenizer = BertTokenizer.from_pretrained(module_name_base, token=access_token)

    data_collator = DefaultDataCollator()

    record_id_train, query1_train, query2_train, labels_train = data('train')
    record_id_test, query1_test, query2_test, labels_test = data('validation')

    train_dataset = Dataset(
        record_id_train, query1_train, query2_train, labels_train,
        tokenizer=tokenizer,
        max_length=data_args.max_length
    )
    dev_dataset = Dataset(
        record_id_test, query1_test, query2_test, labels_test,
        tokenizer=tokenizer,
        max_length=data_args.max_length
    )





    # 加载预训练模型
    # pretrained = BertModel.from_pretrained(module_name_base, token=access_token)
    # pretrained.to(device)
    model = BertForSequenceClassification.from_pretrained(data_args.model_dir, num_labels=3)
    model.to(device)

    for param in model.base_model.parameters():
        param.requires_grad = False


    best_steps, best_metric = train(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        data_collator=data_collator
    )

    print(f'Training Finished! Best step - {best_steps} - Best accuracy {best_metric}')

if __name__ == '__main__':
    import os

    # os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
    main()