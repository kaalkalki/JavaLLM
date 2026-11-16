
package com.javallm.nn;

import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Applies an affine projection along the final input dimension.
 *
 * <p>Weights use shape {@code [outputFeatures, inputFeatures]}; an optional
 * rank-1 bias is broadcast across leading input dimensions. Rank-1 inputs return
 * a rank-1 output, while higher-rank leading dimensions are preserved.</p>
 */
public final class Linear implements Layer {

    private final Tensor weights;
    private final Tensor bias;

    public Linear(Tensor weights, Tensor bias) {

        Objects.requireNonNull(weights, "Weights cannot be null");
        if (weights.rank() != 2) {
            throw new IllegalArgumentException("Linear weights must be a rank-2 tensor");
        }
        if (bias != null) {
            if (bias.rank() != 1) {
                throw new IllegalArgumentException("Linear bias must be a rank-1 tensor");
            }
            if (bias.dimension(0) != weights.dimension(0)) {
                throw new IllegalArgumentException("Bias dimension must match output features");
            }
        }

        this.weights = weights;
        this.bias = bias;
    }

    public Linear(Tensor weights) {
        this(weights, null);
    }

    public Tensor weights() {
        return weights;
    }

    public Tensor bias() {
        return bias;
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        if (input.rank() < 1) {
            throw new IllegalArgumentException("Linear input must have rank >= 1");
        }

        int inputFeatures = input.dimension(input.rank() - 1);
        int expectedFeatures = weights.dimension(1);

        if (inputFeatures != expectedFeatures) {
            throw new IllegalArgumentException("Input features " + inputFeatures + " do not match weight features " + expectedFeatures);
        }

        boolean vectorInput = input.rank() == 1;
        Tensor matrixInput = vectorInput ? input.reshape(1, inputFeatures) : input;
        Tensor output = matrixInput.matmul(weights.transpose());
        if (bias != null) {
            output = output.add(bias);
        }

        if (vectorInput) {
            output = output.reshape(weights.dimension(0));
        }

        return output;
    }
}
