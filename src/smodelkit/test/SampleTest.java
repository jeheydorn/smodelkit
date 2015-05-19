package smodelkit.test;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.math3.util.ArithmeticUtils;
import org.junit.Test;

import smodelkit.Matrix;
import smodelkit.util.IntegerListComparator;
import smodelkit.util.Pair;
import smodelkit.util.Range;
import smodelkit.util.Sample;

public class SampleTest
{

	@Test
	public void sampleWithReplacement100PercentTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'simple_nominal -c -2'\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE x3   {up, down}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	{tasty,not-tasty}\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,8.88,up,pig,tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"red,0.112,up,pig,not-tasty\n" + 
				"\n" + 
				"\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Matrix[] actual = Sample.sampleWithReplacement(new Random(), inputs, labels, 1.0);
		assertEquals(4, actual[0].rows());
		assertEquals(4, actual[1].rows());
		assertEquals(3, actual[0].cols());
		assertEquals(2, actual[1].cols());
	}

	@Test
	public void sampleWithReplacement0PercentTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'simple_nominal -c -2'\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE x3   {up, down}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	{tasty,not-tasty}\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,8.88,up,pig,tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"red,0.112,up,pig,not-tasty\n" + 
				"\n" + 
				"\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Matrix[] actual = Sample.sampleWithReplacement(new Random(), inputs, labels, 0.0);
		assertEquals(0, actual[0].rows());
		assertEquals(0, actual[1].rows());
		assertEquals(3, actual[0].cols());
		assertEquals(2, actual[1].cols());
	}

	@Test
	public void sampleWithReplacement20PercentTest()
	{
		Matrix data = new Matrix();
		data.loadFromArffString("@RELATION 'simple_nominal -c -2'\n" + 
				"\n" + 
				"@ATTRIBUTE x1	{blue,red,green}\n" + 
				"@ATTRIBUTE x2	real\n" + 
				"@ATTRIBUTE x3   {up, down}\n" + 
				"@ATTRIBUTE class1	{pig,sheep}\n" + 
				"@ATTRIBUTE class2	{tasty,not-tasty}\n" + 
				"\n" + 
				"@DATA\n" + 
				"blue,8.88,up,pig,tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"blue,8.88,down,sheep,not-tasty\n" + 
				"red,0.112,up,pig,not-tasty\n" + 
				"\n" + 
				"\n" + 
				"\n" + 
				"");
		Pair<Matrix> inputsAndLabels = data.splitInputsAndLabels();
		Matrix inputs = inputsAndLabels.getFirst();
		Matrix labels = inputsAndLabels.getSecond();
		Matrix[] actual = Sample.sampleWithReplacement(new Random(), inputs, labels, 0.2);
		assertEquals(1, actual[0].rows());
		assertEquals(1, actual[1].rows());
		assertEquals(3, actual[0].cols());
		assertEquals(2, actual[1].cols());
	}
	
	@Test
	public void sampleWithoutReplacementTest()
	{
		Random rand = new Random(0);
		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 10, 10);
			assertEquals(10, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 9, 10);
			assertEquals(9, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 0, 10);
			assertEquals(0, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 1, 10);
			assertEquals(1, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 2, 10);
			assertEquals(2, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}
		
		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 5, 10);
			assertEquals(5, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 5, 10000000000000000L);
			assertEquals(5, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 1, 1);
			assertEquals(1, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 1, 2);
			assertEquals(1, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}

		{
			Collection<Long> actual = Sample.sampleWithoutReplacement(rand, 0, 0);
			assertEquals(0, actual.size());
			assertEquals(new HashSet<>(actual).size(), actual.size());
		}
	}
	
	@Test
	public void samplePermutationsWithoutReplacementTest()
	{
		Random rand = new Random(0);
		{
			for (int seqLength : new Range(1, 6))
			{
				for (int i : new Range((int)ArithmeticUtils.factorial(seqLength) + 1))
				{
					Collection<List<Integer>> actual = Sample.samplePermutationsWithoutReplacement(rand,
							i, seqLength);
					assertEquals(i, actual.size());
					testPermutation(actual);		
				}
			}
		}
		
		{
			Collection<List<Integer>> actual = Sample.samplePermutationsWithoutReplacement(rand, 0, 6);
			assertEquals(0, actual.size());
			testPermutation(actual);		
		}
		
		// Test when the total possible permutations is greater than a long can store.
		Collection<List<Integer>> actual = Sample.samplePermutationsWithoutReplacement(rand, 3, 1000);
		assertEquals(3, actual.size());
		testPermutation(actual);
		
		// Make sure an exception is thrown if more samples than possible permutations are requested.
		try
		{
			Sample.samplePermutationsWithoutReplacement(rand, (int)ArithmeticUtils.factorial(10) + 1, 10);
		}
		catch(IllegalArgumentException e)
		{
			// Success.
			return;
		}
		fail();
	}
	
	private void testPermutation(Collection<List<Integer>> permutations)
	{
		// Check for duplicates.
		Set<List<Integer>> set = new TreeSet<List<Integer>>(new IntegerListComparator());
		set.addAll(permutations);
		assertEquals(set.size(), permutations.size());		
		
		int expectedSize = permutations.size() > 0 ? permutations.iterator().next().size() : 0;
		
		// Make sure all indexes are present in each permutation.
		for (List<Integer> perm : permutations)
		{
			// They should all be the same length.
			assertEquals(expectedSize, perm.size());
			
			Set<Integer> permSet = new HashSet<>(perm);
			
			for (int i : new Range(perm.size()))
			{
				assertTrue(permSet.contains(i));
			}
		}

	}
	
	@Test
	public void privateMethodTest()
	{
		Sample.testPrivateMethods();
	}

}
