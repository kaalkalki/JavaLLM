package com.javallm.transformer;

import com.javallm.nn.Normalization;
import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Llama pre-normalized decoder block with attention and dense SwiGLU residual paths.
 *
 * <p>It normalizes and applies self-attention, adds the first residual, then
 * normalizes and applies the feed-forward sublayer before adding the second
 * residual. Input and output shapes are required to match.</p>
 */
public final class LlamaTransformerBlock implements TransformerBlock {

    private final Normalization attentionNorm;
    private final Attention attention;
    private final Normalization feedForwardNorm;
    private final FeedForward feedForward;
    private final int hiddenSize;

    public LlamaTransformerBlock(
            Normalization attentionNorm,
            Attention attention,
            Normalization feedForwardNorm,
            FeedForward feedForward
    ) {
        this.attentionNorm = Objects.requireNonNull(attentionNorm, "Attention norm cannot be null");
        this.attention = Objects.requireNonNull(attention, "Attention cannot be null");
        this.feedForwardNorm = Objects.requireNonNull(feedForwardNorm, "Feed-forward norm cannot be null");
        this.feedForward = Objects.requireNonNull(feedForward, "Feed-forward cannot be null");
        this.hiddenSize = attention.hiddenSize();
        if (feedForward.hiddenSize() != hiddenSize
                || attentionNorm.featureSize() != hiddenSize
                || feedForwardNorm.featureSize() != hiddenSize) {
            throw new IllegalArgumentException("Transformer block dimensions are inconsistent");
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        return forwardInternal(input, 0, null, -1);
    }

    public Tensor forwardWithCache(Tensor input, int startPosition, KeyValueCache cache, int layerIndex) {
        return forwardInternal(input, startPosition, cache, layerIndex);
    }

    @Override
    public int hiddenSize() {
        return hiddenSize;
    }

    @Override
    public int cacheHeadCount() {
        return attention.cacheHeadCount();
    }

    @Override
    public int cacheHeadDimension() {
        return attention.cacheHeadDimension();
    }

    private Tensor forwardInternal(Tensor input, int startPosition, KeyValueCache cache, int layerIndex) {
        Objects.requireNonNull(input, "Input cannot be null");
        Tensor normalizedInput = attentionNorm.forward(input);
        Tensor attentionOutput = cache == null
                ? attention.forward(normalizedInput)
                : attention.forwardWithCache(normalizedInput, startPosition, cache, layerIndex);
        requireSameShape(input, attentionOutput, "Attention output");
        Tensor afterAttention = input.add(attentionOutput);

        Tensor feedForwardOutput = feedForward.forward(feedForwardNorm.forward(afterAttention));
        requireSameShape(afterAttention, feedForwardOutput, "Feed-forward output");
        return afterAttention.add(feedForwardOutput);
    }

    private static void requireSameShape(Tensor expected, Tensor actual, String name) {
        int[] expectedShape = expected.shape();
        int[] actualShape = actual.shape();
        if (!java.util.Arrays.equals(expectedShape, actualShape)) {
            throw new IllegalStateException(name + " shape does not match residual shape");
        }
    }
}
