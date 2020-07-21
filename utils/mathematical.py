import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def my_local_extremum(y, min_or_max, do_plot=True):
    """Find all local extremum indexes.
        The output extremum_indexes will hold the locations in y which hold the
        extremums, it will also be sorted from biggest extremum of y to smallest
        extremum of y.
        This function considers a local-maximum to be a point which is either bigger then both its
        neighbours or bigger then or and equals the other. A flat line will not
        be considered a local maximum. The edges ARE being taken in
        consideration. A local-minimum has the analogue definition.

        Args:
            y (numpy array): A series of real numbers
            min_or_max (str): Either "min" or "max", telling which extremums of y you'd like to detect
            do_plot (bool): True for plotting the outcome

        Returns:
            numpy array: A series of indices that correspond to the extremums

        Raises:
            Checks for whether min_or_max is set properly, otherwise it doesn't check the other variables.

        Examples:
            >>> y = np.array([0, 1, 0, 3, 0, 2])
            >>> my_local_extremum(y, min_or_max='max', do_plot=False)
            >>> maxima = my_local_extremum(y, min_or_max='max', do_plot=False)
            >>> print(maxima)
            [3 5 1]
        """
    if min_or_max == 'max':
        y_prime = y
    elif min_or_max == 'min':
        y_prime = -y
    else:
        sys.exit("You must declare what min_or_max of extremums you're looking for, max or min!")
    dy = np.diff(y_prime)

    indicator = np.diff(np.sign(dy))
    extremum_indexes = np.asarray(np.where(indicator < 0)) + 1
    # Check edges:
    if dy[0] < 0:
        extremum_indexes = np.concatenate((np.array([[0]]), extremum_indexes), axis=1)
    if dy[-1] > 0:
        extremum_indexes = np.concatenate((extremum_indexes, np.array([[len(y_prime) - 1]])), axis=1)

    # Sorting the extremums in ascending order:
    extremums = y_prime[extremum_indexes]

    ranked = np.argsort(-extremums)
    ordered_extremum_indexes = extremum_indexes[0][ranked[::-1]]

    if do_plot:
        plt.plot(range(len(y)), y, color='blue', linestyle='-', linewidth=2, label="blue line")
        plt.plot(ordered_extremum_indexes, y[ordered_extremum_indexes], 'ro', markersize=8, label="red dots")
        plt.xlabel("t")
        plt.ylabel("y(t)")
        plt.legend(labels=['Original series', 'Points of ' + min_or_max])
        plt.show()

    return ordered_extremum_indexes[0]
