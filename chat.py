import torch
import torch.nn.functional as F
import os
import tiktoken
from model import GPT, GPTConfig


# load model checkpoint
def load_model(checkpoint_path, device):
    enc = tiktoken.get_encoding("gpt2")

    # create model
    model = GPT(GPTConfig(vocab_size=50304))  # adjust as necessary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    return model, enc


# generate response
def generate_response(
    model, enc, prompt, max_length=50, num_return_sequences=1, device="cpu"
):
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    xgen = tokens.repeat(num_return_sequences, 1)
    sample_rng = torch.Generator(device=device).manual_seed(42)

    with torch.no_grad():
        while xgen.size(1) < max_length:
            logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    return [enc.decode(xgen[i, :].tolist()) for i in range(num_return_sequences)]


# chat loop
def chat(model, enc, device):
    print("Chatbot is ready to chat! Type 'exit' to end the conversation.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        response = generate_response(model, enc, user_input, device=device)
        print(f"Assistant: {response[0]}")


if __name__ == "__main__":
    checkpoint_path = "log/model_19072.pt"  # specify the path to your .pt file
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, enc = load_model(checkpoint_path, device)
    chat(model, enc, device)
