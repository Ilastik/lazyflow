# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Copyright 2011-2014, the ilastik developers

from collections import OrderedDict

class OrderedSignal(object):
    """
    A simple callback mechanism that ensures callbacks occur in the same order as subscription.
    """
    def __init__(self):
        self.callbacks = OrderedDict()

    def subscribe(self, fn, **kwargs):
        """
        Subscribe the given callable to be called when the signal is fired.
        If the callable is already subscribed to the signal, it is relocated to the end of the callback list.
        
        :param fn: The callable to add to this signal's list of callbacks. Must be hashable.
        :param kwargs: **DEPRECATED**.  Additional parameters to include when the signal calls the function. 
                       Instead of using this parameter, consider binding arguments to your callable 
                       with ``functools.partial`` or (better) ``ilastik.bind``.
        """
        # Remove this function if we already have it
        self.unsubscribe(fn)
        # Add it to the end
        self.callbacks[fn] = kwargs

    def unsubscribe(self, fn):
        """
        Unsubscribe the given function from the signal's callback list.
        If the callable was not found in the list, this function returns silently.
        
        :param fn: The callable to remove from the subscription list.
        
        .. note:: This relies on the callable's ``__eq__`` operator.  
                  Note that ``functools.partial`` objects do not implement special support 
                  for ``__eq__``. If your callback is of that type, you must provide the 
                  exact instance when unsubscribing.  Note that ``ilastik.bind`` objects 
                  ARE equality comparable.  For those callables, it is not necessary to 
                  provide the exact instance of the callable that was used for subscription.
                  An equivalent ``ilastik.bind`` object (same target and bound args) will suffice.
        """
        # Find this function and remove its entry
        try:
            self.callbacks.pop(fn)
        except KeyError:
            pass


    def __call__(self, *args):
        """
        Emit the signal.  Calls each callback in the subscription list, in order, with the specified arguments.
        """
        for f, kw in self.callbacks.items():
            f(*args, **kw)

    def clean(self):
        self.callbacks = OrderedDict()
