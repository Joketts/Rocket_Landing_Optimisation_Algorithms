package coursework;

import model.Individual;
import model.Fitness;
import model.NeuralNetwork;


public class SimulatedAnnealing extends NeuralNetwork {
    @Override
    public void run() {

        // Initialize current solution
        Individual current = createRandomIndividual();
        current.fitness = Fitness.evaluate(current, this);
        evaluations++;

        // Best-so-far
        best = current.copy();

        int maxEvals = Parameters.maxEvaluations;
        double mutateRate = Parameters.mutateRate;
        double mutateDelta = Parameters.mutateChange;

        // Temperature setup
        double InitialTemp     = Parameters.maxGene - Parameters.minGene;
        double currentTemp      = InitialTemp;
        double tempStep = InitialTemp / (double)maxEvals;

        // Adaptive cooling window
        int windowSize      = 500;
        int windowCount     = 0;
        int acceptedWindow  = 0;
        double lowRatio     = 0.15;
        double highRatio    = 0.40;

        // Best-of-K sampling
        int K = 5;

        // Polishing schedule
        int polishEvery = 2000;
        int polishMoves = 600;

        // Main SA loop
        while (evaluations < maxEvals) {
            // Linear cooling step
            currentTemp -= tempStep;
            if (currentTemp < 1e-8) currentTemp = 1e-8;

            // Best-of-K Neighbor Sampling
            Individual bestCand = null;
            for (int k = 0; k < K && evaluations < maxEvals; k++) {
                Individual cand = current.copy();
                // jitter each gene
                for (int i = 0; i < cand.chromosome.length; i++) {
                    if (Parameters.random.nextDouble() < mutateRate) {
                        double val = cand.chromosome[i]
                                + Parameters.random.nextGaussian() * mutateDelta;
                        // clamp
                        val = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, val));
                        cand.chromosome[i] = val;
                    }
                }
                cand.fitness = Fitness.evaluate(cand, this);
                evaluations++;
                if (bestCand == null || cand.fitness < bestCand.fitness) {
                    bestCand = cand;
                }
            }
            if (bestCand == null) break;

            // Metropolis acceptance
            double dE = bestCand.fitness - current.fitness;
            boolean accepted = false;
            if (dE <= 0 || Parameters.random.nextDouble() < Math.exp(-dE / currentTemp)) {
                current = bestCand;
                accepted = true;
                if (current.fitness < best.fitness) {
                    best = current.copy();
                }
            }

            // Update adaptive window
            windowCount++;
            if (accepted) acceptedWindow++;
            if (windowCount >= windowSize) {
                double ratio = (double)acceptedWindow / windowCount;
                if (ratio < lowRatio) {
                    // reheating
                    currentTemp = Math.min(InitialTemp, currentTemp / 0.9);
                } else if (ratio > highRatio) {
                    // faster cooling
                    currentTemp *= 0.95;
                }
                windowCount    = 0;
                acceptedWindow = 0;
            }

            // Periodic polishing via Variable-Neighborhood Descent
            if (evaluations % polishEvery < K) {
                current = greedyDescent(current, polishMoves);
            }
        }

        saveNeuralNetwork();
    }

    // VNDP using
    // small 1 gene small change
    // medium 2 gene medium change
    // large 10% gene large change
    private Individual greedyDescent(Individual seed, int moves) {
        Individual current = seed.copy();
        int totalGenes = current.chromosome.length;
        int perNbr     = moves / 3;
        double deltaSmall  = Parameters.mutateChange * 0.5;
        double deltaMedium = Parameters.mutateChange;
        int    numLarge    = Math.max(1, totalGenes / 10);
        double deltaLarge  = Parameters.mutateChange * 2;

        for (int nbr = 0; nbr < 3; nbr++) {
            int iter = (nbr == 2) ? moves - 2 * perNbr : perNbr;
            for (int m = 0; m < iter && evaluations < Parameters.maxEvaluations; m++) {
                Individual neighbor = current.copy();
                if (nbr == 0) {
                    // small jitter one gene
                    int idx = Parameters.random.nextInt(totalGenes);
                    double val = neighbor.chromosome[idx]
                            + Parameters.random.nextGaussian() * deltaSmall;
                    val = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, val));
                    neighbor.chromosome[idx] = val;
                } else if (nbr == 1) {
                    // medium jitter two genes
                    for (int j = 0; j < 2; j++) {
                        int idx = Parameters.random.nextInt(totalGenes);
                        double val = neighbor.chromosome[idx]
                                + Parameters.random.nextGaussian() * deltaMedium;
                        val = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, val));
                        neighbor.chromosome[idx] = val;
                    }
                } else {
                    // large jitter 10% of genes
                    for (int j = 0; j < numLarge; j++) {
                        int idx = Parameters.random.nextInt(totalGenes);
                        double val = neighbor.chromosome[idx]
                                + Parameters.random.nextGaussian() * deltaLarge;
                        val = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, val));
                        neighbor.chromosome[idx] = val;
                    }
                }
                neighbor.fitness = Fitness.evaluate(neighbor, this);
                evaluations++;
                if (neighbor.fitness < current.fitness) {
                    current = neighbor;
                    if (current.fitness < best.fitness) {
                        best = current.copy();
                    }
                }
            }
        }
        return current;
    }

    private Individual createRandomIndividual() {
        Individual ind = new Individual();
        ind.chromosome = new double[Parameters.getNumGenes()];
        for (int i = 0; i < ind.chromosome.length; i++) {
            ind.chromosome[i] = Parameters.minGene
                    + (Parameters.maxGene - Parameters.minGene) * Parameters.random.nextDouble();
        }
        return ind;
    }

    @Override
    public double activationFunction(double x) {
        if (x < -20) return -1;
        if (x > 20)  return 1;
        return Math.tanh(x);
    }
}

