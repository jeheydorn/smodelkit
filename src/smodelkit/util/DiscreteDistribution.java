package smodelkit.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DiscreteDistribution <T>
{
	private List<T> items;
	private List<Double> weights;
	private Random rand;
	
	public DiscreteDistribution(List<T> items, List<Double> weights, Random rand)
	{
		if (items.size() == 0)
			throw new IllegalArgumentException();
		if (items.size() != weights.size())
			throw new IllegalArgumentException();
		this.items = items;
		this.weights = weights;
		this.rand = rand;
	}
	
	public T sample()
	{
		return items.get(sampleIndex());
	}
	
	public List<T> sampleManyTimes(int size)
	{
		List<T> result = new ArrayList<T>(size);
		for (int i = 0; i < size; i++)
			result.add(sample());
		return result;
	}
	
	/**
	 *  Randomly samples a discrete distribution. The sum of the given weights will be
	 *  essentially normalized to ensure a probability distribution.
	 * @return The index of a weight chosen by sampling.
	 */
	private int sampleIndex()
	{
        double totalWeight = 0;
        for (Double weight : weights)
        	totalWeight += weight;
        
        double uniformSample = rand.nextDouble();
        uniformSample *= totalWeight;
        
        double acc = 0;
        int i = 0;
        while (true)
        {
            acc += weights.get(i);
            if (acc >= uniformSample)
                break;
            i += 1;
        }
        return i;
	}
}
