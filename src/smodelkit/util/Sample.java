package smodelkit.util;

import static org.junit.Assert.*;

import java.util.*;
import java.util.stream.Collectors;

import org.apache.commons.math3.util.ArithmeticUtils;
import org.apache.commons.math3.util.MathUtils;

import smodelkit.Matrix;
import smodelkit.Vector;

/**
 * Helper functions for generating samples.
 * @author joseph
 *
 */
public class Sample
{
	/**
	 * Samples a given dataset with replacement. The size of the result is specified by a
	 * percent of the size of the original dataset.
	 * @param inputs
	 * @param labels
	 * @param percent
	 * @return The first element is the resulting inputs. The second element is the resulting labels.
	 */
	public static Matrix[] sampleWithReplacement(Random rand, Matrix inputs, Matrix labels, double percent)
	{
		if (inputs.rows() != labels.rows())
		{
			throw new IllegalArgumentException();
		}
		
		Matrix[] result = new Matrix[2];
		result[0] = new Matrix();
		result[0].copyMetadata(inputs);
		result[1] = new Matrix();
		result[1].copyMetadata(labels);
		
		int resultRows = (int)Math.round(inputs.rows() * percent);
		for (@SuppressWarnings("unused") int i : new Range(resultRows))
		{
			int r = rand.nextInt(inputs.rows());
			result[0].addRow(inputs.row(r));
			result[1].addRow(labels.row(r));
		}
		
		return result;
	}
	
	/**
	 * Like sampleWithReplacement except instead of duplicating instances in the results,
	 * instance weights are used instead of creating duplicates in the results. The percent
	 * is fixed at 100%.
	 * @param rand
	 * @param inputs
	 * @param labels
	 * @return The first element is the resulting inputs. The second element is the resulting labels.
	 */
	public static Matrix[] sampleWithReplacementUsingInstanceWeights(Random rand, Matrix inputs, Matrix labels)
	{
		if (inputs.rows() != labels.rows())
		{
			throw new IllegalArgumentException();
		}
		
		int[] weights = new int[inputs.rows()];
		for (@SuppressWarnings("unused") int i : new Range(weights.length))
		{
			weights[rand.nextInt(weights.length)]++;
		}
		
		Matrix baggedInputs = new Matrix();
		Matrix baggedLabels = new Matrix();
		baggedInputs.copyMetadata(inputs);
		baggedLabels.copyMetadata(labels);
		for (int i : new Range(weights.length))
		{
			if (weights[i] > 0)
			{
				Vector x = new Vector(inputs.row(i));
				x.setWeight(weights[i]);
				baggedInputs.addRow(x);
				Vector y = new Vector(labels.row(i));
				y.setWeight(weights[i]);
				baggedLabels.addRow(y);
			}
		}
		
		return new Matrix[] {baggedInputs, baggedLabels};
	}
	
	/**
	 * Samples from the natural numbers in the range 0 to N exclusive 
	 * without replacement.
	 * 
	 * This implementation is designed to be efficient when sampleSize is small
	 * and N is very large.
	 * 
	 * @param rand
	 * @param sampleSize The number of samples desired.
	 * @param N
	 * @return
	 */
	public static Collection<Long> sampleWithoutReplacement(Random rand, int sampleSize, long N)
	{
		if (sampleSize > N)
			throw new IllegalArgumentException("sampleSize must be less than or equal to N.");
		
		if (sampleSize > N/2)
		{
			// To be more efficient, generates samples that will not be in the result, 
			// then return the compliment of those samples.
			Collection<Long> compliment = sampleWithoutReplacement(rand, (int)(N - sampleSize), N);
			List<Long> result = new ArrayList<>((int)(sampleSize));
			for (long i = 0; i < N; i++)
			{
				if (!compliment.contains(i))
					result.add(i);
			}
			
			return result;
		}
		else
		{
			Set<Long> result = new HashSet<>(sampleSize);
			while(result.size() < sampleSize)
			{
				result.add(Math.abs(rand.nextLong()) % N);
			}
			
			return result;
		}
	}
	
	/**
	 * Generates sampleSize samples of permutations without replacement.
	 */
	public static Collection<List<Integer>> samplePermutationsWithoutReplacement(Random rand, 
			int sampleSize, int N)
	{
		if (N < 20 && sampleSize > ArithmeticUtils.factorial(N))
			throw new IllegalArgumentException("sampleSize is larger than the number of possible permutations.");
			
		if (N <= 20)
		{
			Collection<Long> sampleIndexes = sampleWithoutReplacement(rand, sampleSize, 
					ArithmeticUtils.factorial(N));
			List<List<Integer>> result = new ArrayList<>();
			for (long index : sampleIndexes)
			{
				long[] perm = ithPermutation(N, index);
				List<Integer> permList = new ArrayList<>(N);
				for (int i : new Range(perm.length))
				{
					permList.add((int) perm[i]);
				}
				result.add(permList);
			}
			return result;
		}
		else
		{
			// ArithmeticUtils.factorial throw an exception because N is too big,
			// so I cannot use it. Instead just randomly generate permutations and
			// throw away duplicates. This will still be efficient even though I am
			// using rejection sampling because there are N! permutations, so random
			// permutations will amost always be unique. 
			
			Set<List<Integer>> result = new TreeSet<>(new IntegerListComparator());
			while(result.size() < sampleSize)
			{
				List<Integer> permList = new Range(N).toList();
				Collections.shuffle(permList, rand);
				result.add(permList);
			}
			return result;
		}
	}
	
	
	/**
	 * Finds the i'th permutation of n items. From my experience this only works for n up to
	 * 20 inclusive. After that it looks like an overflow error occurs.
	 * 
	 * Ported from code at http://stackoverflow.com/questions/7918806/finding-n-th-permutation-without-computing-others
	 * @return An array of indexes for the items in the permutation.
	 */
	private static long[] ithPermutation(int n, long i)
	{
	   int j, k = 0;
	   long[] fact =  new long[n];
	   long[] perm = new long[n];

	   // compute factorial numbers
	   fact[k] = 1;
	   while (++k < n)
	      fact[k] = fact[k - 1] * k;

	   // compute factorial code
	   for (k = 0; k < n; ++k)
	   {
	      perm[k] = i / fact[n - 1 - k];
	      i = i % fact[n - 1 - k];
	   }

	   // readjust values to obtain the permutation
	   // start from the end and check if preceding values are lower
	   for (k = n - 1; k > 0; --k)
	      for (j = k - 1; j >= 0; --j)
	         if (perm[j] <= perm[k])
	            perm[k]++;

	   return perm;
	}
	
	public static void testPrivateMethods()
	{	
		int n = 20;
		long[] expected = new long[20];
		for (int i : new Range(20))
		{
			expected[i] = 19 - i;
		}
		assertArrayEquals(expected, ithPermutation(n, ArithmeticUtils.factorial(n) - 1));
	}
	
}
