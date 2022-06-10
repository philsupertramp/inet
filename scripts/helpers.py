"""
Collection of helper functions used in script files.
"""
import os
import shutil
import sys
from threading import Thread
from typing import Any, List


def decision(message: str) -> bool:
    """Helper to ask for decision input"""
    yes_no = input(f'{message} [y|N] > ')
    return yes_no.lower() in ['y', 'j']


def move_files(files: List[str], target_dir: str) -> List[str]:
    """Moves files stored via file names in iterable into target directory"""

    new_files = []
    for file in files:
        file_name = file.split('/')[-1]
        target = os.path.join(target_dir, file_name)
        shutil.move(file, target)
        new_files.append(target)

    return new_files


class ProgressBar:
    """
    Helper class to send progress bar data to stdout.

    Example:
        >>> from time import sleep
        >>> pb = ProgressBar(10)
        >>> for i in range(10):
        ...     pb.step(i)
        ...     sleep(1)
        ...     if i == 9:
        ...         pb.done()
        [====================] 100%

    """
    def __init__(self, max_num_elements: int, size: int = 20):
        """
        :param max_num_elements: maximum number elements expected
        :param size: size of the progress bar, i.e. the max number of '#' symbols
        """
        ## Maximum number of elements in progress bar
        self.max_num_elements = max_num_elements
        ## size of progress bar (in characters)
        self.size = size

    def step(self, index: int) -> None:
        """
        Executes a step in the progress bar, this can change the output or
        reapply previous version, depending on the current index
        :param index: current index
        """
        j = (index + 1) / self.max_num_elements
        sys.stdout.write('\r')
        sys.stdout.write(f"[{'=' * int(self.size * j):{self.size}}] {int(100 * j)}%")
        sys.stdout.flush()

    @staticmethod
    def done() -> None:
        """
        call this to finalize the progress bar
        """
        sys.stdout.write('\n')
        sys.stdout.flush()


class ThreadWithReturnValue(Thread):
    """
    Custom thread class with return values
    found at https://stackoverflow.com/a/6894023/6904543
    """
    def __init__(self, group=None, target=None, name=None, args=None, kwargs=None, Verbose=None):
        """

        :param group: thread group
        :param target: target method to call
        :param name: give the thread a verbose name
        :param args: args to pass to `target`
        :param kwargs: kwargs to pass to `target`
        :param Verbose: [invalid parameter] verbosity setting
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        super().__init__(group, target, name, args, kwargs)
        ## Storage for return value
        self._return = None

    def run(self) -> None:
        """Overwritten Thread.run method"""
        try:
            if self._target:
                self._return = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def join(self, *args) -> Any:
        """Join with return value"""
        super().join(*args)
        return self._return
