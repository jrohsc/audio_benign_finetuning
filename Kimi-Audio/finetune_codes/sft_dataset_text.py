"""
Text-only dataset for Kimi-Audio fine-tuning.

This is a simplified version of sft_dataset.py that works with TEXT-ONLY conversations
(no audio tokens required). Used for Text BFT experiments.

The key differences from sft_dataset.py:
1. No whisper model required (no audio feature extraction)
2. No audio_tokens field in messages
3. All messages use message_type="text"
4. audio_loss_mask is always zeros (only text loss is computed)
"""

from torch.utils.data import Dataset
from functools import lru_cache
import torch
from typing import Dict, List
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.utils.data import KimiAContent


class LazyTextOnlyDataset(Dataset):
    """Dataset for text-only supervised fine-tuning (no audio)."""

    def __init__(self, raw_data_list, text_tokenizer, max_len: int, kimia_token_offset: int):
        """
        Initialize text-only dataset.

        Args:
            raw_data_list: List of conversation dicts from JSONL
            text_tokenizer: Tokenizer for text encoding
            max_len: Maximum sequence length
            kimia_token_offset: Token offset for Kimi-A special tokens
        """
        super(LazyTextOnlyDataset, self).__init__()
        self.max_len = max_len

        print(f"[TextOnly] There are {len(raw_data_list)} samples in the dataset")

        print(f"[TextOnly] Loading text tokenizer")
        self.text_tokenizer = text_tokenizer

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

        self.pad_token = self.extra_tokens.pad
        self.kimia_token_offset = kimia_token_offset
        self.raw_data = raw_data_list

        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def _tokenize_text(self, text):
        if text is None:
            return None
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        output_type: str = "text",
    ):
        """Tokenize a single text message (no audio support)."""
        kimia_content_msg = KimiAContent()

        role = message["role"]
        has_loss = role == "assistant"

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_assistant_msg_start
                )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"role: {role}")

        message_type = message.get("message_type", "text")

        if message_type == "text":
            text = message["content"]
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.text_extend(text_tokens, has_loss)
            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * len(text_tokens)
            )

            if role == "assistant":
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss)  # eos for text stream
                kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

        elif message_type is None:
            # Empty message (used for assistant start marker)
            pass
        else:
            raise ValueError(f"[TextOnly] Unsupported message_type: {message_type}. "
                           f"Text-only dataset only supports 'text' messages.")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert (
            kimia_content_msg.is_valid()
        ), f"kimia_content_msg is not valid: {kimia_content_msg}"

        return kimia_content_msg

    def tokenize_conversation(
        self, messages: List[Dict], output_type: str = "text", add_assistant_start_msg: bool = True
    ) -> KimiAContent:
        """
        Tokenize a full conversation of text messages.

        Args:
            messages: List of message dicts with role, message_type, content
            output_type: Always "text" for text-only dataset
            add_assistant_start_msg: Whether to add assistant start marker at end
        """
        assert output_type in ["text", "both"]

        msgs: List[KimiAContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None
        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            if previous_role is None:
                tokenize_role = True
            else:
                if message["role"] == previous_role:
                    tokenize_role = False
                else:
                    tokenize_role = True

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                output_type=output_type,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                message={
                    "role": "assistant",
                    "message_type": None,
                },
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
            )
            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]

        for msg in msgs[1:]:
            ret_msg.merge(msg)

        return ret_msg

    @lru_cache(maxsize=None)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """Get a single text-only sample."""
        task_type = self.raw_data[i].get("task_type", "understanding")
        conversation = self.raw_data[i]["conversation"]

        # For text-only, always use "text" output type
        output_type = "text"

        tokenized_conversation = self.tokenize_conversation(
            conversation, output_type=output_type, add_assistant_start_msg=False
        )

        audio_input_ids, text_input_ids, is_continuous_mask, audio_token_loss_mask, text_token_loss_mask = tokenized_conversation.to_tensor()

        # No audio features for text-only
        audio_features = []

        audio_labels = torch.cat((audio_input_ids[:, 1:], audio_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        text_labels = torch.cat((text_input_ids[:, 1:], text_input_ids.new_full((1, 1), self.pad_token)), dim=1)
        audio_loss_mask = torch.cat((audio_token_loss_mask[:, 1:], audio_token_loss_mask.new_full((1, 1), False)), dim=1)
        text_loss_mask = torch.cat((text_token_loss_mask[:, 1:], text_token_loss_mask.new_full((1, 1), False)), dim=1)

        ret = dict(
            input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=audio_features,  # Empty for text-only
            is_continuous_mask=is_continuous_mask,
            labels=(
                audio_labels,
                text_labels,
                audio_loss_mask,
                text_loss_mask,
            ),
        )

        return ret

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1, "micro batch size is 1 for demo"
        return batch[0]
