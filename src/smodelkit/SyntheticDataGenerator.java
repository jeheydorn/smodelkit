package smodelkit;

import static java.lang.System.out;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import smodelkit.util.DiscreteDistribution;
import smodelkit.util.DoubleArrayComparator;
import smodelkit.util.Helper;
import smodelkit.util.ListMap;
import smodelkit.util.Range;

public class SyntheticDataGenerator
{	
	public static void generateData() throws FileNotFoundException
	{
		Random rand = new Random();
		String resultsFilename = "Datasets/mdc/synthetic/continuous_4out_2-4-8-16classes.arff";
		
		int numNoisyOutputColumns = 0;
		int noisyClasses = 0;
		
		boolean continuousInputs = true; // false means nominal inputs.
		double inputNoiseStandardDeviation = 0.1;
		double inputNoiseProbability = 0.1;
		double outputNoiseProbability = 0.1;
		// Whether the outputs are dependent upon previous outputs.
		Class<? extends Generator> generatorType = LimitedOutputsGenerator.class;
		// If LimitedOutputsGenerator or TwoPartitionLimitedOutputsGenerator are used, then this must be <= numOutputVectors.
		int maxOutputVectorsPerInput = 3;
		int resultRows = 2000;
		int numInputVectors = 20;
		// This is the total number of valid output vectors. Having this should create unconditional dependencies
		// between outputs. This is not used with IndependentGenerator.
		int numOutputVectors = 20;
		
		// Load a matrix with the correct meta-data.
		Matrix metaData = new Matrix();
		if (continuousInputs)
		{
			metaData.loadFromArffFile("Datasets/mdc/synthetic/continuous_meta.arff");
		}
		else
		{
			metaData.loadFromArffFile("Datasets/mdc/synthetic/nominal_meta.arff");
		}
		
		if (numNoisyOutputColumns > 0)
		{
			// Split the inputs from the labels.
			Matrix inputsTmp = new Matrix(metaData, 0, 0, metaData.rows(), metaData.cols() - metaData.getNumLabelColumns());
			Matrix labelsTmp = new Matrix(metaData, 0, metaData.cols() - metaData.getNumLabelColumns(), metaData.rows(),
					metaData.getNumLabelColumns());

			for (int i = 0; i < numNoisyOutputColumns; i++)
			{
				Matrix noiseColumn = new Matrix();
				String arff = String.format("@RELATION 'synthetic_data -c %d'\n" + 
						"\n" + 
						"@ATTRIBUTE noise%d	{", metaData.getNumLabelColumns(), i+1);
				
				Iterator<Integer> it = new Range(noisyClasses);
				while (it.hasNext())
				{
					int n = it.next();
					arff += "n" + (n + 1);
					if (it.hasNext())
						arff += ",";
				}
				
				arff += "}\n"
						+ "@DATA";
						
				noiseColumn.loadFromArffString(arff);


				inputsTmp.copyColumns(noiseColumn, 0, 1);
			}
			for (int c : new Range(labelsTmp.cols()))
				inputsTmp.copyColumns(labelsTmp, c, 1);
			int originalNumLabelColumns = metaData.getNumLabelColumns();
			metaData = inputsTmp;
			metaData.setNumLabelColumns(originalNumLabelColumns + numNoisyOutputColumns);
		}
				
		// Generate each input randomly.
		List<double[]> inputs = new ArrayList<>();
		for (@SuppressWarnings("unused") int i : new Range(numInputVectors))
		{
			double[] input = new double[metaData.cols() - metaData.getNumLabelColumns()];
			for (int j : new Range(input.length))
			{
				if (continuousInputs)
				{
					input[j] = rand.nextGaussian();
				}
				else
				{
					input[j] = rand.nextInt(metaData.getValueCount(j));
				}
			}
			inputs.add(input);
		}
		
		List<Double> inputWeights = new ArrayList<>();
		for (@SuppressWarnings("unused") int i : new Range(inputs.size()))
		{
			inputWeights.add(1.0/inputs.size());
		}
		
		DiscreteDistribution<double[]> inputDist = new DiscreteDistribution<double[]>(
				inputs, inputWeights, rand);
		

		Generator genFunc = null;
		if (generatorType.equals(IndependentGenerator.class))
		{
			genFunc = new IndependentGenerator(rand);
		}
		else if (generatorType.equals(LimitedOutputsGenerator.class))
		{
			genFunc = new LimitedOutputsGenerator(rand, maxOutputVectorsPerInput, numOutputVectors, metaData);
		}
		else if (generatorType.equals(TwoPartitionLimitedOutputsGenerator.class))
		{
			genFunc = new TwoPartitionLimitedOutputsGenerator(rand, maxOutputVectorsPerInput, numOutputVectors, metaData);
		}
						
		// Generate the rows.
		for (int r = 0; r < resultRows; r++)
		{
			double[] input = inputDist.sample();
			double[] outputs = genFunc.generate(input, metaData);
			
			double[] row = Helper.concatArrays(input, outputs);
			// Add noise.
			if (continuousInputs)
				row = addContinuousNoise(row, inputNoiseStandardDeviation, 0, inputs.get(0).length, metaData, rand);
			else
				row = addNominalNoise(row, inputNoiseProbability, 0, inputs.get(0).length, metaData, rand);
			row = addNominalNoise(row, outputNoiseProbability, inputs.get(0).length, row.length, metaData, rand);
			
			if (numNoisyOutputColumns > 0)
				row = makeIrrelevantColumns(row, metaData, rand);
			
			metaData.addRow(Vector.create(row));
		}
		
		out.println("Writing results to: " + resultsFilename);
		try(PrintWriter w = new PrintWriter(resultsFilename))
		{
			if (continuousInputs)
				w.println("% inputNoiseStandardDeviation: " + inputNoiseStandardDeviation);	
			else
				w.println("% inputNoiseProbability: " + inputNoiseProbability);
			w.println("% numNoisyOutputColumns: " + numNoisyOutputColumns);
			w.println("% outputNoiseProbability: " + outputNoiseProbability);
			w.println("% generator type: " + generatorType);
			if (generatorType.equals(LimitedOutputsGenerator.class)
					|| generatorType.equals(TwoPartitionLimitedOutputsGenerator.class))
				w.println("% maxOutputVectorsPerInput: " + maxOutputVectorsPerInput);
			w.println("% resultRows: " + resultRows);
			w.println("% Number of inputs before adding noise (numInputsVectors): " + inputs.size());
			if (generatorType.equals(LimitedOutputsGenerator.class)
					|| generatorType.equals(TwoPartitionLimitedOutputsGenerator.class))
				w.println("% numOutputVectors = " + numOutputVectors);
			w.println("% numInputVectors: " + numInputVectors);
			w.println(metaData.toString());
		}
	}
		
	interface Generator
	{
		/**
		 * Generates an output vector for the given inputs. The input is stored so that if this
		 * method is called again, the result will be from the same distributions.
		 * @param inputs
		 * @param metaData This is needed to tell what label values can be generated.
		 * @return
		 */
		public abstract double[] generate(double[] inputs, Matrix metaData);

	}
	
	private static class LimitedOutputsGenerator implements Generator
	{
		Random rand;
		Map<double[], DiscreteDistribution<double[]>> outputDistributions;
		int maxOutputVectorsPerInput;
		List<double[]> outputs;

		protected LimitedOutputsGenerator(Random rand, int maxOutputVectorsPerInput, int numOutputVectors,
				Matrix metaData)
		{ 
			this.rand = rand;
			this.maxOutputVectorsPerInput = maxOutputVectorsPerInput;
			outputDistributions = new TreeMap<double[], DiscreteDistribution<double[]>>(new DoubleArrayComparator());
			
			if (maxOutputVectorsPerInput > numOutputVectors)
				throw new IllegalArgumentException();
			
			outputs = new ArrayList<>();
			for (@SuppressWarnings("unused") int i : new Range(numOutputVectors))
			{
				double[] o = new double[metaData.getNumLabelColumns()];
				for (int j : new Range(o.length))
				{
					int colIndex = j + metaData.cols() - metaData.getNumLabelColumns();
					o[j] = rand.nextInt(metaData.getValueCount(colIndex));
				}
				outputs.add(o);
			}
		}

		@Override
		public double[] generate(double[] inputs, Matrix metaData)
		{
			DiscreteDistribution<double[]> outputDist = outputDistributions.get(inputs);	
			if (outputDist == null)
			{				
				// Randomly choose how many outputs this input can map to.
				int numOutputVectors = rand.nextInt(maxOutputVectorsPerInput) + 1;
				
				// Each output vector has a random weight.
				List<Double> weights = new ArrayList<Double>();
				for (int i = 0; i < numOutputVectors; i++)
					weights.add(rand.nextDouble());


				List<double[]> outputVectors = new ArrayList<double[]>();
				List<double[]> outputsCopy = new LinkedList<>(outputs);
				for (int i = 0; i < numOutputVectors; i++)
				{
					// Sample from the valid outputs without replacement.
					int index = rand.nextInt(outputsCopy.size());
					double[] outputVector = outputsCopy.get(index);
					outputVectors.add(outputVector);
					outputsCopy.remove(index);
				}
											
				outputDist = new DiscreteDistribution<double[]>(outputVectors, weights, rand);
				outputDistributions.put(inputs, outputDist);
			}
			
			return outputDist.sample();
		}
		
	}

	private static class TwoPartitionLimitedOutputsGenerator implements Generator
	{
		// Each sub-list is a list of all possible valid (ignoring noise) output vectors for a partition.
		List<List<double[]>> outputsByPartition;
		Random rand;
		// Each element is a map from inputs to a distribution over a piece of a label coresponding to a partition.
		List<Map<double[], DiscreteDistribution<double[]>>> outputDistributionsByPartition;
		int maxOutputVectorsPerInput;


		protected TwoPartitionLimitedOutputsGenerator(Random rand, int maxOutputVectorsPerInput, int numOutputVectors,
				Matrix metaData)
		{ 
			this.rand = rand;
			if (maxOutputVectorsPerInput > numOutputVectors)
				throw new IllegalArgumentException();
			this.maxOutputVectorsPerInput = maxOutputVectorsPerInput;
			
			outputsByPartition = new ArrayList<>();
			outputsByPartition.add(new ArrayList<>());
			outputsByPartition.add(new ArrayList<>());
			
			// Create all valid (ignoring noise) output columns for partition 0.
			int p1Columns = metaData.getNumLabelColumns()/2;
			for (@SuppressWarnings("unused") int i : new Range(numOutputVectors))
			{
				double[] o = new double[p1Columns];
				for (int j : new Range(o.length))
				{
					int colIndex = j + metaData.cols() - metaData.getNumLabelColumns();
					o[j] = rand.nextInt(metaData.getValueCount(colIndex));
				}
				outputsByPartition.get(0).add(o);
			}

			// Create all valid (ignoring noise) output columns for partition 1.
			int p2Columns = metaData.getNumLabelColumns() - p1Columns;
			for (@SuppressWarnings("unused") int i : new Range(numOutputVectors))
			{
				double[] o = new double[p2Columns];
				for (int j : new Range(o.length))
				{
					int colIndex = j + metaData.cols() - metaData.getNumLabelColumns() + p1Columns;
					o[j] = rand.nextInt(metaData.getValueCount(colIndex));
				}
				outputsByPartition.get(1).add(o);
			}
			
			outputDistributionsByPartition = new ArrayList<>();
			outputDistributionsByPartition.add(new TreeMap<>(new DoubleArrayComparator()));
			outputDistributionsByPartition.add(new TreeMap<>(new DoubleArrayComparator()));
		}

		@Override
		public double[] generate(double[] inputs, Matrix metaData)
		{
			double[] result = new double[0];
			for (int partitionNumber : new Range(2))
			{
				DiscreteDistribution<double[]> partitionDist = outputDistributionsByPartition.get(partitionNumber).get(inputs);	
				if (partitionDist == null)
				{				
					// Randomly choose how many outputs this input can map to.
					int numOutputVectors = rand.nextInt(maxOutputVectorsPerInput) + 1;
					
					// Each output vector has a random weight.
					List<Double> weights = new ArrayList<Double>();
					for (int i = 0; i < numOutputVectors; i++)
						weights.add(rand.nextDouble());
	
	
					List<double[]> outputVectors = new ArrayList<double[]>();
					List<double[]> outputsCopy = new LinkedList<>(outputsByPartition.get(partitionNumber));
					for (int i = 0; i < numOutputVectors; i++)
					{
						// Sample from the valid outputs without replacement.
						int index = rand.nextInt(outputsCopy.size());
						double[] outputVector = outputsCopy.get(index);
						outputVectors.add(outputVector);
						outputsCopy.remove(index);
					}
												
					partitionDist = new DiscreteDistribution<double[]>(outputVectors, weights, rand);
					outputDistributionsByPartition.get(partitionNumber).put(inputs, partitionDist);
				}
				result = Helper.concatArrays(result, partitionDist.sample());
			}
			
			return result;
		}
		
	}
	

	private static class IndependentGenerator implements Generator
	{		
		Random rand;
		// Maps from inputs to lists of distributes, where the n'th element of the list is the 
		// distribution for the n'th output.
		ListMap<double[], DiscreteDistribution<Double>> outputDistributions;

		public IndependentGenerator(Random rand)
		{
			this.rand = rand;
			outputDistributions = new ListMap<double[], DiscreteDistribution<Double>>(new DoubleArrayComparator());
		}
		
		@Override
		public double[] generate(double[] input, Matrix metaData)
		{
			// Given the input, every output is generated independently.
			
			List<DiscreteDistribution<Double>> distForInput = outputDistributions.get(input);
			if (distForInput == null)
			{
				for (int c : new Range(input.length, metaData.cols()))
				{	
					// Randomly choose how many outputs this input can map to.
					int numOutputValues = rand.nextInt(metaData.getValueCount(c)) + 1;
					
					List<Integer> options = Helper.iteratorToList(new Range(metaData.getValueCount(c)));
	
					// Sample from options without replacement.
					List<Double> outputValues = new ArrayList<>();
					for (int ignored = 0; ignored < numOutputValues; ignored++)
					{
						int i = rand.nextInt(options.size());
						outputValues.add((double)(int)options.get(i));
						options.remove(i);
					}
								
					List<Double> weights = new ArrayList<Double>();
					for (int i = 0; i < outputValues.size(); i++)
						weights.add(rand.nextDouble());
					
					DiscreteDistribution<Double> outputDist = new DiscreteDistribution<>(outputValues, weights, rand);
					outputDistributions.add(input, outputDist);
				}
			}
			
			distForInput = outputDistributions.get(input);
			double[] result = new double[metaData.getNumLabelColumns()];
			for (int c : new Range(metaData.getNumLabelColumns()))
			{
				result[c] = distForInput.get(c).sample();
			}			
			
			return result;
		}
	}
			
	/**
	 * Randomly adds noise to an array of samples from nominal distributions, only in the column between startCol
	 * (inclusive) and endCol (exclusive).
	 * @param in The samples from nominal distributions.
	 * @param noiseProbability The probability of adding noise to an element of in.
	 * @param data Used to keep nominal values within the correct range.
	 * @param rand
	 */
	private static double[] addNominalNoise(double[] in, double noiseProbability, int startCol, int endCol, 
			Matrix data, Random rand)
	{
        assert noiseProbability <= 1.00001 && noiseProbability >= -0.00001;
		double[] result = new double[in.length];
		for (int i = 0; i < result.length; i++)
		{
			result[i] = in[i];
			if (i < startCol || i >= endCol)
				continue;
			if (rand.nextFloat() <= noiseProbability)
			{
//				result[i] = rand.nextInt(data.valueCount(i));
				if (rand.nextBoolean())
				{
					result[i] += 1;
					if (result[i] > data.getValueCount(i) - 1)
						result[i] = 0;
				}
				else
				{
					result [i] -= 1;
					if (result[i] < 0)
						result[i] = data.getValueCount(i) - 1;
				}
			}
		}
		return result;
	}	
	
	private static double[] addContinuousNoise(double[] in, double noiseStandardDeviation, int startCol, int endCol, 
			Matrix data, Random rand)
	{
		double[] result = new double[in.length];
		
		for (int i : new Range(in.length))
		{		
			result[i] = in[i];
			if (i < startCol || i >= endCol)
				continue;
			
			result[i] += rand.nextGaussian() * noiseStandardDeviation;
		}
		
		return result;
	}
	
	/**
	 * Overwrites the first numColumns entries in "in" with uniform random samples, returning
	 * the result in a new array.
	 * @param in
	 * @param numColumns
	 * @return
	 */
	private static double[] makeIrrelevantColumns(double[] in, Matrix metaData, Random r)
	{
		double[] result = new double[in.length];
		for (int i : new Range(in.length))
		{
			if (metaData.getAttrName(i).startsWith("noise"))
			{
				result[i] = r.nextInt(metaData.getValueCount(i));
			}
			else
				result[i] = in[i];
		}
		return result;
	}
	
	public static void main(String[] args) throws FileNotFoundException
	{
		generateData();
		out.println("Done.");
	}
}
