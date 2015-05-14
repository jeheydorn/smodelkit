package smodelkit;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import smodelkit.evaluator.Evaluator;
import smodelkit.util.Helper;

/**
 * Maps evaluator types to evaluation results.
 * @author joseph
 *
 */
public class Evaluation
{
	Map<Class<? extends Evaluator>, List<Double>> evaluations;
	Map<Class<? extends Evaluator>, List<ConfusionMatrix>> confusions;
	
	public Evaluation()
	{
		evaluations = new HashMap<>();
		confusions = new HashMap<>();
	}
	
	public void putScores(Class<? extends Evaluator> evaluatorType, List<Double> results)
	{
		evaluations.put(evaluatorType, results);
	}
	
	public List<Double> getScores(Class<? extends Evaluator> evaluatorType)
	{
		return evaluations.get(evaluatorType);
	}
	
	public void putConfusions(Class<? extends Evaluator> evaluatorType, List<ConfusionMatrix> confusions)
	{
		this.confusions.put(evaluatorType, confusions);
	}
	
	/**
	 * Get the confusion matrices (if any exists) for the given evaluator, or null if none exist.
	 * Not all Evaluators create confusion matrices.
	 */
	public List<ConfusionMatrix> getConfusions(Class<? extends Evaluator> evaluatorType)
	{
		return confusions.get(evaluatorType);
	}

	public Set<Class<? extends Evaluator>> getEvaluatorTypes()
	{
		return evaluations.keySet();
	}
	
	public String toStringFormatedDoubles()
	{
		StringBuilder b = new StringBuilder();

		// Convert evaluations to a map keyed by strings to allow alphabetical sorting.
		Map<String, List<Double>> evalStrs = new TreeMap<>();
		for (Map.Entry<Class<? extends Evaluator>, List<Double>> entry : evaluations.entrySet())
		{
			evalStrs.put(entry.getKey().getSimpleName(), entry.getValue());
		}
		
		Iterator<Map.Entry<String, List<Double>>> iter = evalStrs.entrySet().iterator();
		while (iter.hasNext())
		{
			Map.Entry<String, List<Double>> entry = iter.next();
			b.append(entry.getKey());
			b.append("=");
			b.append(Helper.formatDoubleList(entry.getValue()));
			if (iter.hasNext())
				b.append(", ");
		}
		return b.toString();
		
	}
	
	@Override
	public String toString()
	{
		return toStringFormatedDoubles();
	}
}
