package smodelkit.util;
/**
 * Provides a thread safe way to tell how many threads are running to avoid creating too many of them.
 * @author joseph
 *
 */
public class ThreadCounter
{
	private static int count;
	/**
	 * If the number of threads being used is greater than this number, subsequent requests for
	 * threads will give back only 1.
	 */
	private static int maxThreads;
	public static synchronized Integer getMaxThreads()
	{
		return maxThreads;
	}
	public static synchronized void setMaxThreads(int maxThreads)
	{
		ThreadCounter.maxThreads = maxThreads;
		maxThreadsSetByUser = true;
	}
	private static boolean maxThreadsSetByUser;
	public static boolean isMaxThreadsSetByUser()
	{
		return maxThreadsSetByUser;
	}
	
	static
	{
		maxThreadsSetByUser = false;
		count = 0;
		maxThreads = Runtime.getRuntime().availableProcessors();
	}
	
	public static synchronized void freeThreadCount(int threadCount)
	{
		count -= threadCount;
		if (count < 0)
			throw new IllegalStateException("You freed more threads than you reserved.");
	}
	
	/**
	 * Returns the number of threads on this system that this program is not currently using.
	 * The result is constrained to be no greater than MaxThreads.
	 * @return
	 */
	private static int getAvailableThreads()
	{
		int result = maxThreads - count;
		assert result >= 0;
		return result;
	}
	
	/**
	 * Reserve the desired number of threads. It the counter runs out, 0 will be returned. 
	 * Remember to call freeThreadCount() when done with those threads, passing in the
	 * number of threads reserved. 
	 * @param countDesired Must be at least 1.
	 */
	public static synchronized int reserveThreadCount(int countDesired)
	{
		if (countDesired < 1)
			throw new IllegalArgumentException("Cannot reserve less than 1 thread.");
		int result = getAvailableThreads();
		result = Math.min(result, countDesired);
		assert result >= 0;
		count += result;
		return result;
	}
	
	/**
	 * This is the same as calling reserveThreadCount with any number greater than or equal to the
	 *  number of cores on the system.
	 */
	public static synchronized int reserveAllThreads()
	{
		return reserveThreadCount(maxThreads);
	}
}