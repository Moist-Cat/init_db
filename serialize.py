import json
import random

from movieapp.api import Client

from build import get_files, load_file

data = get_files()


client = Client()

actors = set()

for file in data:
    entry = load_file(file)
    title = entry["title"]
    release_year = int(entry["release_year"])
    movie = client.get_or_create_movie(
        title=title,
        release_year=release_year,
        runtime=int(entry["runtime"]),
        rating=entry["rating"]
    )
    client.session.commit()
    for star in set(entry["actors"]):
        if star in actors:
            continue
        star = star.strip()
        actors.add(star)
        tag = client.get_or_create_tag(name=star, category="actor")
        client.session.commit()
        client.create_tagged_movie(movie_id=movie.id, tag_id=tag.id)
        client.session.commit()

    for star in set(entry["staff"]):
        if star in actors:
            continue
        star = star.strip()
        actors.add(star)
        tag = client.get_or_create_tag(name=star, category="staff")
        client.session.commit()
        client.create_tagged_movie(movie_id=movie.id, tag_id=tag.id)
        client.session.commit()

    cover = entry.get("cover", None)
    if cover:
        client.create_link(movie=movie, url=cover, label="Cover")

    for tag_name in set(entry["tags"]):
        category = "genre"
        if tag_name in {"max", "amazon", "netflix"}:
            category = "streaming"
        tag = client.get_or_create_tag(name=tag_name.lower(), category="genre")
        client.session.commit()
        client.create_tagged_movie(movie_id=movie.id, tag_id=tag.id)
        client.session.commit()
    client.session.commit()
