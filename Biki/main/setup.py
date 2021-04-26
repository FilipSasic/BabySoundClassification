import sys
from pathlib import Path


ROOT = Path().absolute()
sys.path += [str(ROOT), str(ROOT.parent)]

for path in [ROOT, ROOT.parent]:

    new_path = path.joinpath('src')
    sys.path += [str(new_path)]
