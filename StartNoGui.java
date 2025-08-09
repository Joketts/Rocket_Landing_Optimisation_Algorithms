package coursework;

import model.Fitness;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class StartNoGui {

	private static final int RUNS = 10;
	private static final boolean DO_CSV          = true;
	private static final boolean DO_DESCRIPTIVE  = true;
	private static final boolean DO_NORMALITY_JB = true;

	public static void main(String[] args) throws IOException {

		Parameters.maxEvaluations = 20_000;
		Parameters.popSize        = 250;
		Parameters.setHidden(3);
		Parameters.mutateRate     = 0.08;
		Parameters.mutateChange   = 0.25;

		double[] fitnesses = new double[RUNS];


		try (PrintWriter out = new PrintWriter(new FileWriter("resultsAntColonyOptimisation.txt"))) {
			if (DO_CSV) {
				out.println("run,trainFitness,testFitness");
				for (int run = 1; run <= RUNS; run++) {
					long seed = System.currentTimeMillis();
					Parameters.seed   = seed;
					Parameters.random = new java.util.Random(seed);

					Parameters.setDataSet(DataSet.Training);
					NeuralNetwork aco = new AntColonyOptimization();
					aco.run();
					double trainFit = Fitness.evaluate(aco);

					Parameters.setDataSet(DataSet.Test);
					double testFit = Fitness.evaluate(aco);

					fitnesses[run - 1] = testFit;
					out.printf("%d,%.4f,%.4f%n", run, trainFit, testFit);
				}
				out.println();
			}

			if (DO_DESCRIPTIVE) {
				double sum = 0;
				for (double v : fitnesses) sum += v;
				double mean = sum / RUNS;

				double sumSq = 0;
				for (double v : fitnesses) sumSq += (v - mean) * (v - mean);
				double sd = Math.sqrt(sumSq / (RUNS - 1));

				double[] sorted = Arrays.copyOf(fitnesses, RUNS);
				Arrays.sort(sorted);
				java.util.function.DoubleUnaryOperator quantile = p -> {
					double pos = p * (RUNS + 1);
					if (pos <= 1)         return sorted[0];
					else if (pos >= RUNS) return sorted[RUNS - 1];
					int lo = (int)Math.floor(pos) - 1;
					int hi = lo + 1;
					double frac = pos - Math.floor(pos);
					return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
				};
				double q1  = quantile.applyAsDouble(0.25);
				double med = quantile.applyAsDouble(0.50);
				double q3  = quantile.applyAsDouble(0.75);
				double iqr = q3 - q1;

				out.println("=== Descriptive statistics (Test) ===");
				out.printf("Mean       = %.4f%n", mean);
				out.printf("SD (n–1)   = %.4f%n", sd);
				out.printf("Q1  (25%%)  = %.4f%n", q1);
				out.printf("Median(50%%)= %.4f%n", med);
				out.printf("Q3  (75%%)  = %.4f%n", q3);
				out.printf("IQR        = %.4f%n", iqr);
				out.println();
			}

			if (DO_NORMALITY_JB) {
				double sum = 0;
				for (double v : fitnesses) sum += v;
				double mean = sum / RUNS;
				double sumSq = 0;
				for (double v : fitnesses) sumSq += (v - mean) * (v - mean);
				double sd = Math.sqrt(sumSq / (RUNS - 1));

				double m3 = 0, m4 = 0;
				for (double v : fitnesses) {
					double d = v - mean;
					m3 += d*d*d;
					m4 += d*d*d*d;
				}
				m3 /= RUNS;
				m4 /= RUNS;
				double skewness = m3 / (sd*sd*sd);
				double kurtosis = m4 / (sd*sd*sd*sd);

				double jb = RUNS/6.0 * (skewness*skewness
						+ 0.25*(kurtosis - 3)*(kurtosis - 3));

				out.println("=== Jarque–Bera normality test (Test) ===");
				out.printf("Skewness = %.4f%n", skewness);
				out.printf("Kurtosis = %.4f%n", kurtosis);
				out.printf("JB-stat  = %.4f%n", jb);
				out.println("(Compare JB to χ²(2)=5.99 for α=0.05)");
			}
		}

		System.out.println("All results saved to resultsAntColonyOptimisation.txt");
	}
}

