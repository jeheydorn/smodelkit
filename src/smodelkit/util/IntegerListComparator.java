package smodelkit.util;

import java.util.Comparator;
import java.util.List;

public class IntegerListComparator implements Comparator<List<Integer>>
{

	@Override
	public int compare(List<Integer> o1, List<Integer> o2)
	{
		for (int i = 0; i < o1.size() && i < o2.size(); i++)
		{				
			if (o1.get(i) < o2.get(i))
				return -1;
			if (o1.get(i) > o2.get(i))
				return 1;
		}
		
		return Integer.compare(o1.size(), o2.size());
	}

}
