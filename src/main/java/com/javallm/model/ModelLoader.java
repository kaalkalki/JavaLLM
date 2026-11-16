package com.javallm.model;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Loads a model and its tokenizer from a model artifact.
 *
 * <p>Format-specific loaders implement this contract so callers can choose a
 * weight format without coupling inference entry points to a parser.</p>
 */
public interface ModelLoader {

    LoadedModel loadModel(Path path) throws IOException;
}
