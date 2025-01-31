from pathlib import Path
import os

import build

dir = Path("metadata_bk_genres_few_movies")

data = build.load_json(dir)

genres = {"History", "Music", "Musical", "Sport", "War", "Documentary", "Western"}

_50k_gang = {"History", "Music", "Musical", "Sport"}
_35k_gang = {"War", "Documentary", "Western"}

remove = {"Film-Noir",} 

for id, movie in data.items():
    if not genres.intersection(data[id]["tags"]):
        Path(str(dir) + "/" + id + ".json").unlink()
    for genre in data[id]["tags"].copy():
        if genre not in genres:
            data[id]["tags"].remove(genre)
    build.save_file(id, data[id], directory=dir, force=True)
