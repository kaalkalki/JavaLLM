package com.javallm;

import com.javallm.gguf.GgufModelLoader;
import com.javallm.inference.GenerationConfig;
import com.javallm.inference.GenerationResult;
import com.javallm.inference.TokenGenerator;
import com.javallm.model.LoadedModel;
import java.nio.file.Path;
import java.util.SplittableRandom;

/**
 * Command-line entry point for loading a GGUF model and generating text.
 *
 * <p>Arguments are the model path, maximum number of generated tokens, and prompt
 * words. The generated continuation is written to standard output.</p>
 */
public final class Main {

    private Main() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java com.javallm.Main <model.gguf> <max-new-tokens> <prompt>");
            System.exit(2);
        }

        Path modelPath = Path.of(args[0]);
        int maximumNewTokens = Integer.parseInt(args[1]);
        StringBuilder promptBuilder = new StringBuilder();
        for (int index = 2; index < args.length; index++) {
            if (index > 2) {
                promptBuilder.append(' ');
            }
            promptBuilder.append(args[index]);
        }

        String prompt = promptBuilder.toString();

        LoadedModel model = GgufModelLoader.load(modelPath);
        GenerationConfig config = new GenerationConfig(maximumNewTokens, 0.8, 40, 0.95);
        TokenGenerator generator = new TokenGenerator(model, new SplittableRandom(1));
        GenerationResult result = generator.generate(prompt, config);
        System.out.print(result.text());
    }
}
