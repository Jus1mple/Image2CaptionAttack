#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QFormer+OPT Model for Feature Leakage Attack.

This module implements the attack model that reconstructs image captions
from leaked intermediate features using QFormer and OPT language model.

The model consists of:
1. Optional convolutional module to process spatial features
2. Intermediate projection layer (maps leaked features to QFormer input space)
3. BLIP-2 QFormer (for vision-language alignment)
4. Language projection layer (maps QFormer output to language model space)
5. OPT language model with LoRA adaptation

Architecture:
    Leaked Features → [Conv Module] → Projection → QFormer → LM Projection → OPT → Caption
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import (
    Blip2Config,
    Blip2QFormerModel,
    Blip2ForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model


# ==================== Task Definitions ====================

class ModelTask:
    """Supported tasks for the QFormer+OPT model."""
    CAPTION = "caption"
    OCR = "ocr"  # Note: OCR is not fully implemented yet
    # Future tasks could include:
    # DETECTION = "detection"
    # SEGMENTATION = "segmentation"
    # CLASSIFICATION = "classification"


# ==================== Model Output Class ====================

class Blip2ForConditionalGenerationModelOutput:
    """
    Output class for QFormerOptModel forward pass.

    This class wraps the outputs from different model components for easier
    access and manipulation.

    Attributes:
        loss: Cross-entropy loss for caption generation (if labels provided)
        logits: Language model output logits
        vision_outputs: Processed intermediate features
        qformer_outputs: QFormer output embeddings
        language_model_outputs: Full language model outputs
    """

    def __init__(
        self,
        loss: Optional[torch.FloatTensor] = None,
        logits: Optional[torch.FloatTensor] = None,
        vision_outputs: Optional[torch.FloatTensor] = None,
        qformer_outputs: Optional[torch.FloatTensor] = None,
        language_model_outputs: Optional[torch.FloatTensor] = None,
    ):
        self.loss = loss
        self.logits = logits
        self.vision_outputs = vision_outputs
        self.qformer_outputs = qformer_outputs
        self.language_model_outputs = language_model_outputs

    def to_tuple(self):
        """Convert outputs to tuple format."""
        return (
            self.loss,
            self.logits,
            self.vision_outputs,
            self.qformer_outputs,
            self.language_model_outputs,
        )

    def to_dict(self):
        """Convert outputs to dictionary format."""
        return {
            "loss": self.loss,
            "logits": self.logits,
            "vision_outputs": self.vision_outputs,
            "qformer_outputs": self.qformer_outputs,
            "language_model_outputs": self.language_model_outputs,
        }

    def __getitem__(self, item):
        """Enable indexing like a tuple."""
        return self.to_tuple()[item]

    def __len__(self):
        """Return number of output components."""
        return len(self.to_tuple())

    def __iter__(self):
        """Enable iteration over outputs."""
        return iter(self.to_tuple())

    def __repr__(self):
        """String representation of outputs."""
        return str(self.to_dict())

    def __str__(self):
        """String representation of outputs."""
        return str(self.to_dict())


# ==================== Main Model ====================

class QFormerOptModel(nn.Module):
    """
    QFormer+OPT model for reconstructing captions from leaked features.

    This model takes intermediate features from a victim vision model and
    generates image captions using a QFormer encoder and OPT decoder.

    The attack pipeline:
    1. Extract intermediate features from victim model (e.g., ResNet layer2)
    2. Optionally process spatial features with convolutional module
    3. Project features to QFormer input space
    4. Use QFormer to align features with language space
    5. Generate captions using OPT language model with LoRA

    Args:
        blip2model: Pre-trained BLIP-2 model (source of QFormer, projections, and LM)
        blip2vison_output_dim: Output dimension of BLIP-2 vision encoder
            Default: 1408 (for BLIP-2 ViT)
        intermediate_feature_dim: Dimension of leaked intermediate features
            Default: 512 (can be different based on victim model and layer)
        extra_conv_module: Optional convolutional module for processing spatial features
            Used when leaked features have spatial dimensions (e.g., [C, H, W])

    Example:
        >>> from transformers import Blip2ForConditionalGeneration
        >>> blip2 = Blip2ForConditionalGeneration.from_pretrained("blip2-opt-2.7b")
        >>> model = QFormerOptModel(
        ...     blip2model=blip2,
        ...     blip2vison_output_dim=1408,
        ...     intermediate_feature_dim=512,
        ...     extra_conv_module=None
        ... )
        >>> model.freeze_all_except_target_layers()
        >>> outputs = model(
        ...     task="caption",
        ...     intermediate_features=leaked_features,
        ...     input_ids=caption_input_ids,
        ...     labels=caption_labels
        ... )
    """

    def __init__(
        self,
        blip2model: Blip2ForConditionalGeneration,
        blip2vison_output_dim: int = 1408,
        intermediate_feature_dim: int = 512,
        extra_conv_module: Optional[nn.Module] = None,
    ):
        super(QFormerOptModel, self).__init__()

        # Store BLIP-2 configuration
        self.config: Blip2Config = blip2model.config

        # Optional convolutional module for spatial feature processing
        self.extra_conv_module = extra_conv_module

        # Linear projection: leaked features → QFormer input space
        self.intermediate_projection = nn.Linear(
            intermediate_feature_dim, blip2vison_output_dim
        )
        nn.init.xavier_uniform_(self.intermediate_projection.weight)

        # QFormer: aligns vision and language representations
        self.qformer: Blip2QFormerModel = blip2model.qformer

        # Trainable query tokens (learnable queries for QFormer)
        self.query_tokens = blip2model.query_tokens

        # Language projection: QFormer output → language model input space
        self.language_projection = blip2model.language_projection

        # Language model (OPT) with LoRA for parameter-efficient fine-tuning
        self.language_model = blip2model.language_model

        # Apply LoRA to language model for efficient adaptation
        peft_config = LoraConfig(
            r=4,  # Low-rank dimension
            lora_alpha=16,  # Scaling factor
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
            lora_dropout=0.1,
            bias="none",
        )
        self.language_model = get_peft_model(self.language_model, peft_config)

    def freeze_all_except_target_layers(
        self,
        target_layers: list = ["intermediate_projection", "qformer", "language_projection"]
    ):
        """
        Freeze all model parameters except specified target layers.

        This is used during training to only update specific components while
        keeping others (like the pre-trained language model) frozen.

        Args:
            target_layers: List of layer name prefixes to keep trainable
                Default: ["intermediate_projection", "qformer", "language_projection"]

        Example:
            >>> model.freeze_all_except_target_layers()  # Freeze LM, train projection + QFormer
            >>> model.freeze_all_except_target_layers(["qformer"])  # Only train QFormer
        """
        for name, param in self.named_parameters():
            # Check if parameter belongs to any target layer
            param.requires_grad = any(target in name for target in target_layers)

        # Always keep extra conv module trainable (if it exists)
        if self.extra_conv_module is not None:
            for param in self.extra_conv_module.parameters():
                param.requires_grad = True

    # ==================== Embedding Accessors ====================

    def get_input_embeddings(self):
        """Get language model input embeddings."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set language model input embeddings."""
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        """Set language model output embeddings."""
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        """Get language model output embeddings."""
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        """Get language model encoder (if encoder-decoder architecture)."""
        return self.language_model.get_encoder()

    def get_decoder(self):
        """Get language model decoder."""
        return self.language_model.get_decoder()

    # ==================== Forward Methods ====================

    def forward(
        self,
        task: str,
        intermediate_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        """
        Forward pass for the QFormer+OPT model.

        Args:
            task: Task to perform ("caption" or "ocr")
            intermediate_features: Leaked intermediate features from victim model
                Shape: [batch_size, feature_dim] or [batch_size, C, H, W]
            input_ids: Input token IDs for prompts/captions
            attention_mask: Attention mask for input_ids
            decoder_input_ids: Decoder input IDs (for encoder-decoder models)
            decoder_attention_mask: Attention mask for decoder
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            labels: Ground truth labels for computing loss
            return_dict: Whether to return ModelOutput object
            interpolate_pos_encoding: Whether to interpolate positional encodings

        Returns:
            Blip2ForConditionalGenerationModelOutput with loss and logits

        Raises:
            ValueError: If task is not supported

        Example:
            >>> outputs = model(
            ...     task="caption",
            ...     intermediate_features=leaked_features,
            ...     input_ids=caption_tokens,
            ...     labels=caption_labels
            ... )
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        if task == ModelTask.CAPTION:
            return self._forward_for_caption(
                intermediate_features,
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                output_attentions,
                output_hidden_states,
                labels,
                return_dict,
                interpolate_pos_encoding,
            )
        elif task == ModelTask.OCR:
            # Note: OCR task uses same implementation as caption for now
            # Future work could add OCR-specific processing
            return self._forward_for_caption(
                intermediate_features,
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                output_attentions,
                output_hidden_states,
                labels,
                return_dict,
                interpolate_pos_encoding,
            )
        else:
            raise ValueError(
                f"Task '{task}' not supported. "
                f"Supported tasks: {ModelTask.CAPTION}, {ModelTask.OCR}"
            )

    def _forward_for_caption(
        self,
        intermediate_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        """
        Forward pass for caption generation task.

        Pipeline:
        1. [Optional] Process spatial features with conv module
        2. Project features to QFormer input space
        3. Run QFormer with query tokens
        4. Project QFormer output to language model space
        5. Concatenate with text embeddings
        6. Generate captions with language model
        7. Compute loss if labels provided

        Args:
            intermediate_features: Leaked features [B, D] or [B, C, H, W]
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]
            labels: Ground truth token IDs for loss computation
            (other args same as forward())

        Returns:
            Model output with loss and logits
        """
        # Step 1: Process spatial features if needed
        if self.extra_conv_module is not None:
            intermediate_features = self.extra_conv_module(intermediate_features)

        # Step 2: Project intermediate features to QFormer hidden size
        intermediate_features = self.intermediate_projection(
            intermediate_features
        ).unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]

        # Step 3: Prepare query tokens and attention mask
        batch_size = intermediate_features.shape[0]
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        image_attention_mask = torch.ones(
            intermediate_features.size()[:-1],
            dtype=torch.long,
            device=intermediate_features.device,
        )

        # Step 4: Pass through QFormer
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=intermediate_features,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = qformer_outputs.last_hidden_state

        # Step 5: Project QFormer output to language model input space
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )

        # Step 6: Get text embeddings and concatenate with vision features
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        input_embeds = torch.cat(
            [language_model_inputs, input_embeds.to(language_model_inputs.device)],
            dim=1,
        )

        # Step 7: Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat(
            [language_model_attention_mask, attention_mask.to(language_model_attention_mask.device)],
            dim=1
        )

        # Step 8: Language model forward pass
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits if return_dict else outputs[0]
        loss = None

        # Step 9: Compute loss if labels provided
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1):, :]

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, self.config.text_config.vocab_size),
                shift_labels.view(-1),
            )

        # Return output
        if not return_dict:
            output = (logits, intermediate_features, query_output, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=intermediate_features,
            qformer_outputs=query_output,
            language_model_outputs=outputs,
        )

    # ==================== Generation Methods ====================

    def generate(
        self,
        task: str,
        intermediate_features: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ):
        """
        Generate captions from intermediate features.

        Args:
            task: Task to perform ("caption" or "ocr")
            intermediate_features: Leaked features from victim model
            input_ids: Optional prompt token IDs
            attention_mask: Optional attention mask for prompts
            interpolate_pos_encoding: Whether to interpolate positional encodings
            **generate_kwargs: Additional arguments for language model generation
                (e.g., max_length, num_beams, temperature, top_k, top_p)

        Returns:
            Generated token sequences

        Example:
            >>> generated_ids = model.generate(
            ...     task="caption",
            ...     intermediate_features=leaked_features,
            ...     max_length=50,
            ...     num_beams=5
            ... )
            >>> captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        """
        if task == ModelTask.CAPTION:
            return self._generate_for_caption(
                intermediate_features,
                input_ids,
                attention_mask,
                interpolate_pos_encoding,
                **generate_kwargs,
            )
        elif task == ModelTask.OCR:
            # OCR uses same generation as caption for now
            return self._generate_for_caption(
                intermediate_features,
                input_ids,
                attention_mask,
                interpolate_pos_encoding,
                **generate_kwargs,
            )
        else:
            raise ValueError(
                f"Task '{task}' not supported. "
                f"Supported tasks: {ModelTask.CAPTION}, {ModelTask.OCR}"
            )

    @torch.no_grad()
    def _generate_for_caption(
        self,
        intermediate_features: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ):
        """
        Generate captions using beam search or greedy decoding.

        This method processes features through QFormer and then uses the
        language model's generation capabilities.

        Args:
            intermediate_features: Leaked features [B, D] or [B, C, H, W]
            input_ids: Optional prompt tokens
            attention_mask: Optional attention mask
            interpolate_pos_encoding: Whether to interpolate positions
            **generate_kwargs: Generation parameters

        Returns:
            Generated token sequences [B, generated_length]
        """
        # Step 1: Process spatial features if needed
        if self.extra_conv_module is not None:
            intermediate_features = self.extra_conv_module(intermediate_features)

        # Step 2: Project intermediate features to QFormer space
        intermediate_features = self.intermediate_projection(
            intermediate_features
        ).unsqueeze(1)

        batch_size = intermediate_features.size(0)

        # Step 3: Prepare QFormer inputs
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        image_attention_mask = torch.ones(
            intermediate_features.size()[:-1],
            dtype=torch.long,
            device=intermediate_features.device,
        )

        # Step 4: QFormer forward pass
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=intermediate_features,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = qformer_outputs.last_hidden_state

        # Step 5: Project to language model space
        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )

        # Step 6: Prepare input tokens (use BOS if no prompt provided)
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(intermediate_features.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Step 7: Concatenate vision and text attention masks
        attention_mask = torch.cat(
            [language_attention_mask, attention_mask.to(language_attention_mask.device)],
            dim=1,
        )

        # Step 8: Get input embeddings and concatenate
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
            dim=1,
        )

        # Step 9: Adjust generation parameters for decoder-only models
        if not self.language_model.config.is_encoder_decoder:
            # Add vision tokens to max_length and min_length
            generate_kwargs["max_length"] = (
                generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            )
            generate_kwargs["min_length"] = (
                generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]
            )

        # Step 10: Generate tokens
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # Step 11: Prepend BOS token for consistency
        if not self.language_model.config.is_encoder_decoder:
            bos_tokens = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(intermediate_features.device)
            )

            if not isinstance(outputs, torch.Tensor):
                # Handle BeamSearchOutput or other output types
                outputs.sequences = torch.cat(
                    [bos_tokens.expand(outputs.sequences.shape[0], -1), outputs.sequences],
                    dim=-1,
                )
            else:
                # Handle tensor output
                outputs = torch.cat([bos_tokens.expand(outputs.shape[0], -1), outputs], dim=-1)

        return outputs


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    print("QFormer+OPT Model for Feature Leakage Attack")
    print("=" * 60)
    print("\nThis module implements the attack model architecture.")
    print("Import and use it in your training/evaluation scripts.")
    print("\nExample:")
    print("  from models.qformer_opt import QFormerOptModel, ModelTask")
    print("  model = QFormerOptModel(blip2, ...)")
    print("  outputs = model(task=ModelTask.CAPTION, ...)")
