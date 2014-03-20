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

from abc import ABCMeta, abstractmethod

def _has_attribute( cls, attr ):
    return True if any(attr in B.__dict__ for B in cls.__mro__) else False

def _has_attributes( cls, attrs ):
    return True if all(_has_attribute(cls, a) for a in attrs) else False    

class DrawableABC:
    __metaclass__ = ABCMeta

    @abstractmethod
    def size(self):
        return NotImplemented

    @abstractmethod
    def drawAt(self, canvas, upperLeft):
        """
        Return the svg text for this item, starting at the given point.
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DrawableABC:
            return True if _has_attributes(C, ['size', 'drawAt']) else False
        return NotImplemented

class ConnectableABC:
    __metaclass__ = ABCMeta

    @abstractmethod
    def key(self):
        return NotImplemented
    
    @abstractmethod
    def partnerKey(self):
        return NotImplemented

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DrawableABC:
            return True if _has_attributes(C, ['key', 'partnerKey']) else False
        return NotImplemented
