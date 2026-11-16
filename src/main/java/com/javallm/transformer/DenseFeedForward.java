package com.javallm.transformer;

import com.javallm.nn.Layer;
import com.javallm.nn.Linear;
import com.javallm.nn.SiLU;
import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Dense SwiGLU feed-forward sublayer used by Llama-style decoder blocks.
 *
 * <p>The gate projection passes through SiLU and is multiplied elementwise by
 * the up projection. The down projection maps the intermediate features back
 * to the model hidden size.</p>
 */
public final class DenseFeedForward implements FeedForward {

    private final Linear gateProjection;
    private final Linear upProjection;
    private final Linear downProjection;
    private final SiLU activation;
    private final int hiddenSize;

    public DenseFeedForward(Linear gateProjection, Linear upProjection, Linear downProjection) {
        this.gateProjection = Objects.requireNonNull(gateProjection, "Gate projection cannot be null");
        this.upProjection = Objects.requireNonNull(upProjection, "Up projection cannot be null");
        this.downProjection = Objects.requireNonNull(downProjection, "Down projection cannot be null");
        Tensor gateWeights = gateProjection.weights();
        Tensor upWeights = upProjection.weights();
        Tensor downWeights = downProjection.weights();
        if (gateWeights.rank() != 2 || upWeights.rank() != 2 || downWeights.rank() != 2
                || gateWeights.dimension(0) != upWeights.dimension(0)
                || gateWeights.dimension(1) != upWeights.dimension(1)
                || downWeights.dimension(1) != gateWeights.dimension(0)
                || downWeights.dimension(0) != gateWeights.dimension(1)) {
            throw new IllegalArgumentException("Feed-forward projection dimensions are inconsistent");
        }
        this.hiddenSize = gateWeights.dimension(1);
        this.activation = new SiLU();
    }

    @Override
    public int hiddenSize() {
        return hiddenSize;
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        if (input.rank() < 1 || input.dimension(input.rank() - 1) != hiddenSize) {
            throw new IllegalArgumentException("Feed-forward input feature size is incorrect");
        }
        Tensor gate = activation.forward(gateProjection.forward(input));
        Tensor up = upProjection.forward(input);
        return downProjection.forward(gate.multiply(up));
    }
}
