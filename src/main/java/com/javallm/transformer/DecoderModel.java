package com.javallm.transformer;

import com.javallm.nn.Embedding;
import com.javallm.nn.Linear;
import com.javallm.nn.Normalization;
import com.javallm.nn.RMSNorm;
import com.javallm.model.LanguageModel;
import com.javallm.tensor.Tensor;

import java.util.List;
import java.util.Objects;

/**
 * Single-batch decoder-only transformer that returns
 * {@code [sequence, vocabulary]} logits.
 *
 * <p>Token embeddings pass through the configured transformer blocks and final
 * normalization before the output projection. The model supports ordinary full-sequence
 * inference and incremental calls backed by a {@link KeyValueCache}.</p>
 */
public final class DecoderModel implements LanguageModel {

    private final Embedding tokenEmbedding;
    private final List<TransformerBlock> blocks;
    private final Normalization finalNorm;
    private final Linear outputProjection;
    private final int vocabularySize;
    private final int hiddenSize;
    private final int keyValueHeads;
    private final int headDimension;

    public DecoderModel(
            Embedding tokenEmbedding,
            List<TransformerBlock> blocks,
            RMSNorm finalNorm,
            Linear outputProjection
    ) {
        this(tokenEmbedding, blocks, (Normalization) finalNorm, outputProjection);
    }

    public DecoderModel(
            Embedding tokenEmbedding,
            List<TransformerBlock> blocks,
            Normalization finalNorm,
            Linear outputProjection
    ) {
        this.tokenEmbedding = Objects.requireNonNull(tokenEmbedding, "Token embedding cannot be null");
        this.blocks = List.copyOf(Objects.requireNonNull(blocks, "Blocks cannot be null"));
        this.finalNorm = Objects.requireNonNull(finalNorm, "Final norm cannot be null");
        this.outputProjection = Objects.requireNonNull(outputProjection, "Output projection cannot be null");
        this.vocabularySize = tokenEmbedding.vocabularySize();
        this.hiddenSize = tokenEmbedding.embeddingSize();
        this.keyValueHeads = this.blocks.isEmpty() ? 0 : this.blocks.get(0).cacheHeadCount();
        this.headDimension = this.blocks.isEmpty() ? 0 : this.blocks.get(0).cacheHeadDimension();
        if (finalNorm.featureSize() != hiddenSize
                || outputProjection.weights().rank() != 2
                || outputProjection.weights().dimension(0) != vocabularySize
                || outputProjection.weights().dimension(1) != hiddenSize) {
            throw new IllegalArgumentException("Decoder model dimensions are inconsistent");
        }
        for (TransformerBlock block : this.blocks) {
            if (block.hiddenSize() != hiddenSize
                    || block.cacheHeadCount() != keyValueHeads
                    || block.cacheHeadDimension() != headDimension) {
                throw new IllegalArgumentException("Decoder blocks use inconsistent attention dimensions");
            }
        }
    }

    @Override
    public int vocabularySize() {
        return vocabularySize;
    }

    @Override
    public int hiddenSize() {
        return hiddenSize;
    }

    @Override
    public int layerCount() {
        return blocks.size();
    }

    @Override
    public KeyValueCache newKeyValueCache(int maximumSequenceLength) {
        if (blocks.isEmpty()) {
            throw new IllegalStateException("A decoder with no transformer blocks has no KV cache shape");
        }
        return new KeyValueCache(blocks.size(), keyValueHeads, headDimension, maximumSequenceLength);
    }

    @Override
    public Tensor forward(int... tokenIds) {
        return forwardInternal(tokenIds, null);
    }

    @Override
    public Tensor forwardWithCache(int[] tokenIds, KeyValueCache cache) {
        Objects.requireNonNull(cache, "KV cache cannot be null");
        return forwardInternal(tokenIds, cache);
    }

    private Tensor forwardInternal(int[] tokenIds, KeyValueCache cache) {
        Objects.requireNonNull(tokenIds, "Token IDs cannot be null");
        if (tokenIds.length == 0) {
            throw new IllegalArgumentException("At least one token is required");
        }
        int startPosition = cache == null ? 0 : cache.sequenceLength();
        if (cache != null && (long) startPosition + tokenIds.length > cache.maximumSequenceLength()) {
            throw new IllegalArgumentException("Input exceeds KV cache context length");
        }

        Tensor hiddenStates = tokenEmbedding.forward(tokenIds).reshape(1, tokenIds.length, hiddenSize);
        for (int layerIndex = 0; layerIndex < blocks.size(); layerIndex++) {
            TransformerBlock block = blocks.get(layerIndex);
            hiddenStates = cache == null
                    ? block.forward(hiddenStates)
                    : block.forwardWithCache(hiddenStates, startPosition, cache, layerIndex);
        }
        if (cache != null) {
            cache.completePrefill(startPosition + tokenIds.length);
        }
        Tensor logits = outputProjection.forward(finalNorm.forward(hiddenStates));
        return logits.reshape(tokenIds.length, vocabularySize);
    }
}
