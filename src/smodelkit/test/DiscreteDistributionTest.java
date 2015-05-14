package smodelkit.test;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import smodelkit.util.DiscreteDistribution;

public class DiscreteDistributionTest
{

	@Test
	public void testSampleManyTimes()
	{
		Random rand = new Random();
		DiscreteDistribution<Integer> dist = new DiscreteDistribution<Integer>(Arrays.asList(0, 1, 2),
				Arrays.asList(0.8, 0.05, 0.15), rand);
		List<Integer> samples = dist.sampleManyTimes(1000000);
		double zeroCount = 0;
		for (Integer i : samples)
			if (i == 0)
				zeroCount++;
		//System.out.println(zeroCount/samples.size());
		assertEquals(0.8, zeroCount/samples.size(), 0.01);
	}

}
