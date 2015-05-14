package smodelkit.util;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

public class Counter <T>
{
	Map<T, Integer> map;
	Comparator<T> comparator;
	int totalCount;
	
	public Counter()
	{
		map = new TreeMap<>();
	}
	
	public Counter(Comparator<T> comparator)
	{
		map = new TreeMap<>(comparator);
		this.comparator = comparator;
	}

	public void increment (T item)
	{
		Integer count = map.get(item);
		if (count == null)
		{
			map.put(item, 1);
		}
		else
		{
			map.put(item, count + 1);
		}
		totalCount++;
	}
	
	public int getCount(T item)
	{
		Integer count = map.get(item);
		if (count == null)
			return 0;
		else
			return count;
	}
	
	/**
	 * Removes all items from this counter whose count is less than
	 * the given threshold.
	 */
	public void removeItemsWithCountBelow(int threshold)
	{
		Set<T> toRemove = comparator == null ? new TreeSet<>() : new TreeSet<>(comparator);
		for (Map.Entry<T, Integer> entry : map.entrySet())
		{
			if (entry.getValue() < threshold)
			{
				totalCount -= entry.getValue();
				toRemove.add(entry.getKey());
			}
		}
		
		for (T item : toRemove)
		{
			map.remove(item);
		}
	}
	
	public int maxCount()
	{
		return Collections.max(map.values());
	}
	
	public T argmax()
	{
		return Helper.argmax(map);
	}
	
	public Set<T> keySet()
	{
		return map.keySet();
	}
	
	public int getTotalCount()
	{
		return totalCount;
	}
	
	/**
	 * Returns a list containing all items in this counter with their associated counts, ordered
	 * from highest count to lowest.
	 */
	public List<Tuple2<T, Integer>> toListFromHighToLow()
	{
		List<Tuple2<T, Integer>> result = map.entrySet().stream().map(entry -> new Tuple2<>(entry.getKey(), entry.getValue())).collect(Collectors.toList());
		result.sort((tuple1, tuple2) -> -Integer.compare(tuple1.getSecond(), tuple2.getSecond()));
		return result;
	}
	
	@Override
	public String toString()
	{
		return map.toString();
	}
}
