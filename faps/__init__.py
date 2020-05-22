from pkg_resources import get_distribution
__version__ = get_distribution('faps').version

from faps.alogsumexp import alogsumexp
# data import and export
from faps.genotypeArray import genotypeArray
from faps.read_genotypes import read_genotypes
from faps.convert_genotypes import convert_genotypes
from faps.read_paternity_array import read_paternity_array
from faps.export_to_colony import export_to_colony
#from write_paternity_array import write_paternity_array
# Simulation
from faps.make_parents import make_parents
from faps.make_offspring import make_offspring
from faps.make_sibships import make_sibships
from faps.make_generation import make_generation
from faps.make_power import make_power
# paternity arrays
from faps.effective_nloci import effective_nloci
from faps.incompatibilities import incompatibilities
from faps.paternityArray import paternityArray
from faps.transition_probability import transition_probability
from faps.paternity_array import paternity_array
from faps.draw_fathers import draw_fathers
# clustering
from faps.pairwise_lik_fullsibs import pairwise_lik_fullsibs
from faps.sibshipCluster import sibshipCluster
from faps.lik_partition import lik_partition
from faps.sibship_clustering import sibship_clustering
from faps.summarise_sires import summarise_sires
# mating events
from faps.matingEvents import matingEvents
from faps.mating_events import mating_events
from faps.mating_summary import mating_summary
