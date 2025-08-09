package coursework;


import model.Fitness;
import model.Individual;
import model.NeuralNetwork;
import java.util.*;


public class AntColonyOptimization extends NeuralNetwork {
    private final int archiveSize = Parameters.popSize; // amount of best solutions to keep set to pop size
    private final int numAnts     = Parameters.popSize;
    private final double rankSpread = 0.40; //controls strength of ranking solutions influences pheromone updates
    private final double initialSpread = 0.6; //Initial pheromone spread across solution
    private final double immigrantRatio = 0.08; //amount of colony replaced by rnd solutions
    private final double mutationRate   = Parameters.mutateRate;
    private final double mutationRange  = Parameters.mutateChange;

    @Override
    public void run() {
        // init achieve with rnd chromosomes
        List<Individual> archive = initialiseArchive();
        best = getBestFromArchive(archive);

        // helper array
        double[] prob = new double[archiveSize];

        // Pre‑compute maximum generations so we can cool xi linearly
        final int maxGen = Parameters.maxEvaluations / numAnts;
        int gen = 0;

        while (evaluations < Parameters.maxEvaluations) {
            gen++;
            // Sort / elitism
            Collections.sort(archive, Comparator.comparingDouble(ind -> ind.fitness));
            if (archive.get(0).fitness < best.fitness) best = archive.get(0).copy();

            // Rank‑based sampling probabilities
            double sum = 0.0;
            for (int i = 0; i < archiveSize; i++) {
                prob[i] = Math.exp(-(i * (double)i) / (2 * (rankSpread * archiveSize) * (rankSpread * archiveSize)));
                sum += prob[i];
            }
            for (int i = 0; i < archiveSize; i++) prob[i] /= sum;

            // CDF for roulette wheel
            double[] cdf = new double[archiveSize];
            double accum = 0;
            for (int i = 0; i < archiveSize; i++) {
                accum += prob[i];
                cdf[i] = accum;
            }

            // Cool currentSpread linearly from xi0 → xi0*0.1 across run
            double currentSpread = initialSpread * (1.0 - 0.85 * gen / maxGen);

            // Generate offspring
            List<Individual> offspring = new ArrayList<>(numAnts);
            int immigrants = (int)Math.ceil(immigrantRatio * numAnts);

            for (int k = 0; k < numAnts; k++) {
                Individual child;

                if (k < immigrants) {
                    // Random immigrant
                    child = createRandomIndividual();
                } else {
                    // ACOR sampling
                    // Select parent by roulette on rank CDF
                    double r = Parameters.random.nextDouble();
                    int idx = Arrays.binarySearch(cdf, r);
                    if (idx < 0) idx = -idx - 1;
                    Individual parent = archive.get(idx);

                    child = new Individual();
                    child.chromosome = new double[parent.chromosome.length];

                    for (int d = 0; d < parent.chromosome.length; d++) {
                        // sigma = mean absolute deviation around parent
                        double sigma = 0.0;
                        for (int j = 0; j < archiveSize; j++) {
                            if (j != idx) sigma += Math.abs(archive.get(j).chromosome[d] - parent.chromosome[d]);
                        }
                        sigma = (sigma / (archiveSize - 1)) * currentSpread;

                        double val = parent.chromosome[d] + sigma * Parameters.random.nextGaussian();

                        //per‑gene mutation lecture operator
                        if (Parameters.random.nextDouble() < mutationRate) {
                            val += (Parameters.random.nextDouble() * 2 - 1) * mutationRange;
                        }

                        // clamp to weight bounds
                        child.chromosome[d] = Math.max(Parameters.minGene, Math.min(Parameters.maxGene, val));
                    }
                }

                child.fitness = Fitness.evaluate(child, this);
                evaluations++;
                offspring.add(child);
            }


            // Merge + truncate (elitism implicit)
            archive.addAll(offspring);
            Collections.sort(archive, Comparator.comparingDouble(ind -> ind.fitness));
            archive = new ArrayList<>(archive.subList(0, archiveSize));
        }

        saveNeuralNetwork();
    }

    // Utility helpers

    private List<Individual> initialiseArchive() {
        List<Individual> list = new ArrayList<>(archiveSize);
        for (int i = 0; i < archiveSize; i++) {
            Individual ind = createRandomIndividual();
            ind.fitness = Fitness.evaluate(ind, this);
            evaluations++;
            list.add(ind);
        }
        return list;
    }

    private Individual createRandomIndividual() {
        Individual ind = new Individual();
        ind.chromosome = new double[Parameters.getNumGenes()];
        for (int d = 0; d < ind.chromosome.length; d++) {
            ind.chromosome[d] = Parameters.minGene +
                    (Parameters.maxGene - Parameters.minGene) * Parameters.random.nextDouble();
        }
        return ind;
    }

    private Individual getBestFromArchive(List<Individual> archive) {
        return archive.stream().min(Comparator.comparingDouble(i -> i.fitness)).get().copy();
    }

    @Override
    public double activationFunction(double x) {
        if (x < -20) return -1;
        if (x > 20)  return 1;
        return Math.tanh(x);
    }
}



