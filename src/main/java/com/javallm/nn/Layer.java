package com.javallm.nn;

import com.javallm.tensor.Tensor;

/**
 * A neural-network component that transforms an input tensor.
 *
 * <p>Layer implementations expose their computation through {@link #forward(Tensor)}.</p>
 */
public interface Layer {
    Tensor forward(Tensor input);
}
