import torch


class NERModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module):
        """Creates an instance of the transformer w/ provided backbone

        Args:
            backbone (torch.nn.Module): _description_
        """
        super().__init__()
        self.backbone = backbone
        self.fc_h1 = torch.nn.Linear(768, 768)
        self.fc_out = torch.nn.Linear(768, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Basic PyTorch forward function

        Args:
            input_ids (torch.Tensor): token ids generated by tokenizer
            attention_mask (torch.Tensor): attention mask from the tokenizer

        Returns:
            torch.Tensor: a [batch_size]-shaped tensor w/ logits
        """
        base_output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # extracts last hidden states of BERT which will be needed for classification
        hidden_state = base_output.hidden_states[-1]

        # [batch_size, seq_len, hidden_size] -> get first seq item (CLS)
        cls_token = hidden_state[:, 0, :]

        # Get the result from the added FC layers
        output = self.fc_out(torch.nn.functional.gelu(self.fc_h1(cls_token)))

        return output[:, 0]
