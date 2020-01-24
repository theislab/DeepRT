import pydicom
import numpy
from pydicom import read_file

dc = read_file("/home/olle/PycharmProjects/LODE/DeepRT/thickness_map_calculation/data/256692_R_20160524/1.3.6.1.4.1.33437.10.4.4089432.13108567473.25811.4.1.dcm")