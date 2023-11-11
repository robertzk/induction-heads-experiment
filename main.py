import random
import torch
import tqdm

from easy_transformer import EasyTransformer
from transformers import AutoTokenizer
from training_interpretability.data import PileLoader
from training_interpretability.model import BasicTransformer, Config
from training_interpretability.tokenizer import GPT2Tokenizer
from training_interpretability.train import Trainer


def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()


def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

def main():
    model_config = Config(
        d_model=(d_model := 768 // 2),
        debug=False,
        layer_norm_epsilon=1e-5,
        d_vocab=50257,
        init_range=0.02,
        n_context=512,
        d_head=64,
        d_mlp=d_model * 2,
        n_heads=d_model // 64,
        n_layers=3,
        device="cuda"
    )

    print("Loading BasicTransformer")
    model = BasicTransformer(model_config).to("cuda")

    print("Loading reference tokenizer and supporting objects")
    tokenizer = GPT2Tokenizer(AutoTokenizer.from_pretrained("gpt2"))

    batch_size = 100
    context_length = model_config.n_context
    data_loader = PileLoader(batch_size, context_length, tokenizer)
    target_tokens = 3_000_000_000
    train_steps = target_tokens // (batch_size * context_length)
    print(f"Training for {train_steps} train steps")

    trainer = Trainer(model, tokenizer, data_loader, train_steps=train_steps, d_vocab=model_config.d_vocab)
    trainer.train()

    model_params_path = "model_params.pt"
    torch.save(model.state_dict(), model_params_path)

    try:
        random_seq = torch.Tensor([random.sample(range(model.cfg.d_vocab), k=20)*2]).type(torch.long).cuda()
        outputs = model(random_seq).argmax(dim=-1)
        pairs = list(zip(random_seq[:, 1:][0].tolist() + [0], outputs[0].tolist()))
        if (matches := sum(int(correct == predicted) for correct, predicted in pairs[random_seq.shape[1] // 2:])) >= random_seq.shape[1] // 2 * 0.75:
            print(f"Induction heads detected! Random sequence duplication predicted at {matches / (random_seq.shape[1] // 2) * 100:.0f}%")
        else:
            print("Induction heads not detected!")

if __name__ == "__main__":
	main()

