package com.javallm.model;

import com.javallm.gguf.GgufFile;

import java.io.IOException;

/**
 * Builds a model implementation from a parsed GGUF file for one architecture.
 */
public interface ModelArchitecture {

    String architectureName();

    LoadedModel loadModel(GgufFile gguf) throws IOException;
}
