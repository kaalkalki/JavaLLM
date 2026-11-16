package com.javallm.gguf;

import com.javallm.model.LoadedModel;
import com.javallm.model.ModelArchitecture;
import com.javallm.model.ModelLoader;
import com.javallm.model.llama.LlamaArchitecture;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Opens GGUF files and delegates model construction by architecture metadata.
 */
public final class GgufModelLoader implements ModelLoader {

    private static final String METADATA_ARCHITECTURE_KEY = "general.architecture";
    private static final String METADATA_MODEL_NAME_KEY = "general.name";
    private static final String METADATA_FILE_TYPE_KEY = "general.file_type";

    private static final Logger LOGGER = Logger.getLogger(GgufModelLoader.class.getName());

    private final Map<String, ModelArchitecture> architectures;

    public GgufModelLoader() {
        ModelArchitecture llamaArchitecture = new LlamaArchitecture();
        this.architectures = Map.of(llamaArchitecture.architectureName(), llamaArchitecture);
    }

    public static LoadedModel load(Path path) throws IOException {
        return new GgufModelLoader().loadModel(path);
    }

    @Override
    public LoadedModel loadModel(Path path) throws IOException {
        Objects.requireNonNull(path, "Path cannot be null");
        long loadStart = System.nanoTime();
        LOGGER.info("Opening GGUF model artifact.");
        try (GgufFile gguf = GgufFile.open(path)) {
            Map<String, Object> metadata = gguf.metadata();
            String architectureName = stringValue(metadata, METADATA_ARCHITECTURE_KEY);
            String modelName = optionalStringValue(metadata, METADATA_MODEL_NAME_KEY, "unnamed model");
            Object fileType = metadata.get(METADATA_FILE_TYPE_KEY);
            LOGGER.log(Level.INFO, "GGUF metadata: name={0}, architecture={1}, ggufFileTypeId={2}.",
                    new Object[]{modelName, architectureName, fileType == null ? "unspecified" : fileType});

            ModelArchitecture architecture = architectures.get(architectureName);
            if (architecture == null) {
                throw new IOException("Unsupported GGUF model architecture: " + architectureName);
            }
            LOGGER.info("Delegating GGUF model construction to " + architecture.getClass().getSimpleName() + ".");
            LoadedModel loadedModel = architecture.loadModel(gguf);
            LOGGER.log(Level.INFO, "GGUF model loaded in {0} ms.", elapsedMilliseconds(loadStart));
            return loadedModel;
        } catch (IOException exception) {
            LOGGER.log(Level.WARNING, "Could not load GGUF model: " + exception.getMessage(), exception);
            throw exception;
        } catch (IllegalArgumentException | ArithmeticException exception) {
            LOGGER.log(Level.WARNING, "Could not load GGUF model: " + exception.getMessage(), exception);
            throw new IOException("Could not load GGUF model: " + exception.getMessage(), exception);
        }
    }

    private static String stringValue(Map<String, Object> metadata, String key) throws IOException {
        Object value = metadata.get(key);
        if (value instanceof String string) {
            return string;
        }
        throw new IOException("Missing GGUF string metadata: " + key);
    }

    private static String optionalStringValue(Map<String, Object> metadata, String key, String fallback) {
        Object value = metadata.get(key);
        return value instanceof String string ? string : fallback;
    }

    private static long elapsedMilliseconds(long startTime) {
        return (System.nanoTime() - startTime) / 1_000_000L;
    }
}
