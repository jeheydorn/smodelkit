package smodelkit.util;

import java.util.Comparator;
import java.util.TreeMap;
import java.util.function.Supplier;

/**
 * A tree map with a factory method for creating new values.
 * If getOrCreate(key) is called and the key is not mapped to a value
 * in the map, then a new mapping is added with that key mapped
 * to a new instance of the value's class.
 * @author joseph
 *
 * @param <K>
 * @param <V>
 */
@SuppressWarnings("serial")
public class MapF <K, V> extends TreeMap<K, V>
{

	public MapF()
	{
	}
	
	public MapF(Comparator<K> comparator)
	{
		super(comparator);
	}
	
	/**
	 * If the given key is mapped to a value in this map, then that
	 * value is returned. If not, then create() is called to make a
	 * new value, then that value is mapped to key and returned.
	 * @param key
	 * @return
	 */
	public V getOrCreate(K key, Supplier<V> createFun)
	{
		V value = get(key);
		if (value == null)
		{
			value = createFun.get();
			put(key, value);
		}
		return value;
		
	}	

}
