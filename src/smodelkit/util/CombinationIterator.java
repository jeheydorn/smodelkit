package smodelkit.util;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Iterates over all combinations of a given set of items, given a number to choose
 * (think n choose m).
 * @author joseph
 *
 * @param <T>
 */
public class CombinationIterator<T> implements Iterable<List<T>>, Iterator<List<T>> 
{
	private List<T> items;
    private boolean finished;
    private int[] indexes;
	public CombinationIterator(List<T> items, int choose)
	{
       if (items == null || items.size() == 0) 
       {
            throw new IllegalArgumentException("items");
       }
       if (choose <= 0 || choose > items.size()) 
       {
            throw new IllegalArgumentException("choose");
       }
       this.items = items;
       this.finished = false;
       this.indexes = new int[choose];
       for (int i = 0; i < indexes.length; i++)
    	   indexes[i] = i;
 	}

	@Override
	public boolean hasNext()
	{
		return !finished;
	}

	@Override
	public List<T> next()
	{      
		if (!hasNext())
			throw new NoSuchElementException();
		
		List<T> result = genNext();

		checkIfFinished();

		// Prepare the indexes for the next call to next().
		if (!finished)
		{
			// Advance the indexes to the next result;
			if (indexes[indexes.length-1] < items.size() - 1)
			{
				indexes[indexes.length-1]++;
			}
			else
			{
				// Find an index that can be moved forward.
				
				int freeI = indexes.length - 2;
				for (; freeI >= 0; freeI--)
				{
					assert freeI >= 0; // checkIfFinished() should have prevented this.
					if (indexes[freeI] < indexes[freeI+1] - 1)
					{
						// This index has room to move forward.
						break;
					}
				}
				indexes[freeI]++;
				// Move the indexes after freeI to be just in front of freeI.
				for (int i = freeI + 1; i < indexes.length; i++)
				{
					indexes[i] = indexes[i-1] + 1;
				}
			}
		}
		
		return result;
	}
	
	/**
	 *  Check if I'm returning the last item.
	 */
	private void checkIfFinished()
	{
		for (int i = 0; i < indexes.length; i++)
		{
			if (indexes[i] != items.size() - 1 - (indexes.length - 1 - i))
				return;
		}
		finished = true;
	}
	
	private List<T> genNext()
	{
		List<T> result = new ArrayList<T>(indexes.length - 1);
		for (int index : indexes)
		{
			result.add(items.get(index));
		}
		return result;
	}

	@Override
	public void remove()
	{
		throw new UnsupportedOperationException();
	}

	@Override
	public Iterator<List<T>> iterator()
	{
		return this;
	}

}
