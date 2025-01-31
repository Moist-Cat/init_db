import csv
from glob import glob
import json
from pathlib import Path
import time

import requests
from bs4 import BeautifulSoup as bs

MOVIES = "title.basics.tsv"
RATINGS = "title.ratings.tsv"
NAMES = "name.basics.tsv"
ACTOR_MOVIES = "title.principals.tsv"


HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

BASE_URL = "https://www.imdb.com/title/{identifier}/"

TYPE = "movie"

METADATA = Path("metadata")
FAILED = Path("failed")
JSON_DIR = METADATA

# id
FIELDS = (None, None, None, "title", None, "release_year", None, "runtime", None)


JUST_URL = "https://apis.justwatch.com/graphql"

JUST_QUERY = """query GetPopularTitles(
  $allowSponsoredRecommendations: SponsoredRecommendationsInput, 
  $country: Country!, 
  $first: Int! = 70, 
  $language: Language!, 
  $after: String, 
  $popularTitlesFilter: TitleFilter, 
  $popularTitlesSortBy: PopularTitlesSorting! = POPULAR, 
  $sortRandomSeed: Int! = 0, 
  $watchNowFilter: WatchNowOfferFilter!, 
  $offset: Int = 0, 
) {
  popularTitles(
    allowSponsoredRecommendations: $allowSponsoredRecommendations
    country: $country
    filter: $popularTitlesFilter
    first: $first
    sortBy: $popularTitlesSortBy
    sortRandomSeed: $sortRandomSeed
    offset: $offset
    after: $after
  ) {
    __typename
    edges {
      cursor
      node {
        ...PopularTitleGraphql
        __typename
      }
      __typename
    }
    pageInfo {
      startCursor
      endCursor
      hasPreviousPage
      hasNextPage
      __typename
    }
    totalCount
  }
}

fragment PopularTitleGraphql on MovieOrShow {
  __typename
  id
  objectId
  objectType
  content(country: $country, language: $language) {
    scoring {
      imdbVotes
      imdbScore
      __typename
    }
    externalIds {
       imdbId
       __typename
    }
    interactions {
      votesNumber
      __typename
    }
  }
  watchNowOffer(country: $country, platform: WEB, filter: $watchNowFilter) {
    ...WatchNowOffer
    __typename
  }
}

fragment WatchNowOffer on Offer {
  id
  standardWebURL
  availableTo
  __typename
}
"""

just_payload = {
  "operationName": "GetPopularTitles",
  "variables": {
    "first": 100,
    "platform": "WEB",
    "popularTitlesSortBy": "IMDB_SCORE",
    "sortRandomSeed": 0,
    "offset": 0,
    "creditsRole": "DIRECTOR",
    "after": None,
    "popularTitlesFilter": {
      "ageCertifications": [],
      "excludeGenres": [],
      "excludeProductionCountries": [],
      "objectTypes": [
        "MOVIE",
        "SHOW",
        "GENERIC_TITLE_LIST",
        "SHOW_SEASON"
      ],
      "productionCountries": [],
      "subgenres": [],
      "genres": [],
      "packages": [
        "nfx"
      ],
      "excludeIrrelevantTitles": False,
      "presentationTypes": [],
      "monetizationTypes": [],
      "searchQuery": ""
    },
    "watchNowFilter": {
      "packages": [
        "nfx"
      ],
      "monetizationTypes": []
    },
    "language": "en",
    "country": "US",
    "allowSponsoredRecommendations": {
      "pageType": "VIEW_POPULAR",
      "placement": "POPULAR_VIEW",
      "language": "en",
      "country": "US",
      "applicationContext": {
        "appID": "3.9.2-webapp#23bc5a5",
        "platform": "webapp",
        "version": "3.9.2",
        "build": "23bc5a5",
        "isTestBuild": False
      },
      "appId": "3.9.2-webapp#23bc5a5",
      "platform": "WEB",
      "supportedFormats": [
        "IMAGE",
        "VIDEO"
      ],
      "supportedObjectTypes": [
        "MOVIE",
        "SHOW",
        "GENERIC_TITLE_LIST",
        "SHOW_SEASON"
      ],
      "alwaysReturnBidID": True,
      "testingModeForceHoldoutGroup": False,
      "testingMode": False
    }
  },
  "query": JUST_QUERY
}

just_headers = {
  'app-version': '3.9.2-web-web',
  'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
}

def load_list():
    with open("list.txt") as file:
        data = file.read()

    return data.splitlines()

def load_file(key, directory=JSON_DIR):
    file = directory / f"{key}.json"
    try:
        with open(file, encoding="utf-8") as jfile:
            data = json.load(jfile)
    except FileNotFoundError as exc:
        raise KeyError(f"{key} was not found") from exc
    except json.decoder.JSONDecodeError as exc:
        print(exc)
        print(f"CRITICAL - Error fetching data from {key}. Ignoring...")
        return {}

    return data

def save_file(filename, data=None, force=False, directory=JSON_DIR):
    file = directory / f"{filename}.json"

    if file.exists() and not force:
        print(f"ERROR - The {file} exists and {force=}")
        return data

    with open(file, "w", encoding="utf-8") as jfile:
        json.dump(data, jfile, ensure_ascii=False, indent="\t")

    return data


def get_files(directory=JSON_DIR) -> list:
    """
    Get full file path
    """
    return list(map(lambda file: Path(file).name[:-5], glob(str(directory  / "*.json"))))

def load_json(directory=JSON_DIR):
    """
    Load data
    """
    data = {}
    files = get_files(directory)
    print(f"INFO - Loading {len(list(files))} files")
    for file in sorted(files):
        game = load_file(file, directory=directory)
        if game:
            data[file] = game
        else:
            print(f"ERROR - No metadata for {file}")
    return data

def save_json(data=None, force=False):
    data = data or {}

    for key, value in data.items():
        save_file(key, value, force)

    return data

def mov(filename):
    LIST = load_list()

    with open(filename) as file:
        rd = csv.reader(file, delimiter="\t", quotechar='"')
        for line in rd:
            id = line[0]
            if (line[1] != TYPE) and (id not in LIST):
                continue

            mov = METADATA / (str(id) + ".json")
            if not mov.exists():
                if id in LIST:
                    save_file(id, data={})
                else:
                    continue

            data = load_file(id)
            for index, field in enumerate(line):
                if FIELDS[index]:
                    data[FIELDS[index]] = field
            data["tags"] = line[-1].strip().split(",")
            if line[-1] == "\\N":
                data["tags"] = []
            with open(mov, "w") as file:
                json.dump(data, file, indent=4)

            print(f"INFO - Saved {id=}")

def is_good_enough(rate, votes, tags):
    rate_min = 5

    tags_50k = {"History", "Music", "Musical", "Sport"}
    tags_35k = {"War", "Documentary", "Western"}
    tags_remove = {"Film-Noir",}

    if tags_remove.intersection(tags):
        return False
    elif tags_50k.intersection(tags):
        votes_min = 50*1000
    elif tags_35k.intersection(tags):
        votes_min = 35*1000
    else:
        votes_min = 100*1000

    if rate <= rate_min or votes < votes_min:
        return False

    return True


def ratings(filename):
    ids = set()
    with open(filename) as file:
        rd = csv.reader(file, delimiter="\t", quotechar='"')
        for line in rd:
            id = line[0]
            if id == "tconst":
                continue
            rate = float(line[1])
            votes = int(line[2])

            mov = METADATA / Path(id + ".json")
            if not mov.exists():
                continue

            with open(mov) as file:
                data = json.load(file)

            if not is_good_enough(rate, votes, data["tags"]):
                print(f"WARNING - Deleting {id}")
                mov.unlink()
                continue

            ids.add(id)

            data["rating"] = rate
            data["votes"] = votes

            with open(mov, "w") as file:
                json.dump(data, file, indent=4)
            del data

    for id in get_files():
        mov = METADATA / Path(id + ".json")
        if id not in ids:
            mov.unlink()

def actors():
    relevant = {}
    actor_keys = {
        "self",
        "actor",
        "actress",
    }
    last = -1
    with open(ACTOR_MOVIES) as file:
        rd = csv.reader(file, delimiter="\t", quotechar='"')
        actors = []
        staff = []
        for line in rd:
            key, _, name_id, category, _, chara = line
            file = JSON_DIR / f"{key}.json"

            if not file.exists():
                continue
            if key != last:
                if last != -1:
                    data = load_file(last)
                    data["actors"] = actors
                    data["staff"] = staff
                    save_file(last, data, force=True)
                actors = []
                staff = []

            if category in actor_keys:
                actors.append(name_id)
            elif category == "archive_footage" or category == "archive_sound":
                pass
            else:
                if chara != r'\N':
                    print(line)
                staff.append(name_id)
            last = key

def fix_and_add_names():
    names = {}
    with open(NAMES) as file:
        rd = csv.reader(file, delimiter="\t", quotechar='"')
        for line in rd:
            key, name, _, _, _, _ = line
            names[key] = name
    
    for file in get_files():
        data = load_file(file)
        if "actors" not in data:
            data["actors"] = []
        if "staff" not in data:
            data["staff"] = []
        actors = []
        for item in data["actors"]:
            actor = names.get(item, None)
            if actor:
                actors.append(actor)
        staff = []
        for item in data["staff"]:
            actor = names.get(item, None)
            if actor:
                staff.append(actor)

        data["actors"] = actors
        data["staff"] = staff


        if data["tags"] == ["\\N"]:
            data["tags"] = []
        if data["runtime"] == "\\N":
            data["runtime"] = 0

        save_file(file, data, force=True)

def covers():
    for file in get_files():
        data = load_file(file)
        url = BASE_URL.format(identifier=file)

        if "cover" in data and data["cover"]:
            continue
        try:
            res = requests.get(url, headers=HEADERS)
            soup = bs(res.text, "html.parser")
            cover = soup.findAll("img")[1]["src"]
            data["cover"] = cover
        except Exception as exc:
            print(f"ERROR - {file}")
            print(exc)
            continue

        print(file, data)

        save_file(file, data, force=True)

def amazon():
    URL = "https://www.amazon.com/s/?url=search-alias%3Dmovies-tv-intl-ship&field-keywords={title}"
    class_ = "puisg-col puisg-col-4-of-12 puisg-col-8-of-16 puisg-col-12-of-20 puisg-col-12-of-24 puis-list-col-right"

    session = requests.session()
    session.headers.update(HEADERS)

    dvds = []
    for index, item in enumerate(get_files()):
        ua = f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{index}.0.0.0 Safari/537.36"
        session.headers["User-Agent"] = ua

        data = load_file(item)
        if "amazon" in data["tags"]:
            continue

        title = data["title"]
        a = data["actors"][0] if data["actors"] else ""
        b = data["staff"][0] if data["staff"] else ""
        query = f"{title} {a} {b}"
        url = URL.format(title=query)
        print(url)

        res, soup = get_data(url, headers=session.headers)
        if not soup:
            continue

        if "Prime Video" not in soup.text:
            print(f"ERROR - No valid links {item}")
            name = str(item) + ".html"
            with open(FAILED / name, "w") as file:
                file.write(res.text)
            continue

        time.sleep(5)
        
        movies = soup.findAll(class_=class_)
        id = item
        print(f"INFO - Getting Amazon link for {id} {url=}")
        mov = METADATA / (str(id) + ".json")
        if "links" not in data:
            data["links"] = []

        movies = movies or []

        flag = False
        dvd_link = None
        for movie in movies:
            pred = any(actor in movie.text for actor in data["actors"]) or any(staff in movie.text for staff in data["staff"])
            assertion = any(actor in movie.text for actor in data["actors"]) and any(staff in movie.text for staff in data["staff"])
            if pred:
                if not assertion:
                    print(f"WARNING - Couldn't assert if this is the actual movie! {id=}")
                link = "https://www.amazon.com" + "/".join(movie.a["href"].split("/")[:-1])
                flag = True
                if "DVD" in movie.text and not dvd_link:
                    flag = False
                    print(f"WARNING - DVD {id}")
                    dvd_link = link
                    continue
                if link not in data["links"]:
                    if isinstance(data["links"], str):
                        data["links"] = [data["links"]]
                    data["links"].append(link)
                data["tags"].append("amazon")
                print(f"INFO - Got valid link for {id}")
                break


        if not flag:
            if dvd_link and not data["links"]:
                print(f"WARNING - Adding dvd link to {id}")
                dvds.append(id)
                data["links"] = [dvd_link]

            print(f"WARNING - Falied to get Amazon link for {id}")
            name = str(id) + ".html"
            with open(FAILED / name, "w") as file:
                file.write(res.text)

        dvd_link = None
        save_file(id, data, force=True)


    with open("dvd.json", "w", encoding="utf-8") as jfile:
        json.dump(dvds, jfile, ensure_ascii=False, indent="\t")


def get_data(url, headers=None):
    headers = headers or {}
    WAIT = 10
    try:
        res = requests.get(url, headers=headers, timeout=30)
        res.raise_for_status()
        time.sleep(WAIT)
    except Exception as exc:
        print(f"ERROR - {exc}")
        
        return (None, None)

    soup = bs(res.text, "html.parser")

    return res, soup


def netflix():
    """
    XXX Deprecated in favour of netflix_just
    """
    BASE_URL = "https://flixwatch.co/"
    URL = BASE_URL + "?s={title}&id={token}"

    count = 0

    session = requests.session()
    session.headers.update(HEADERS)

    for index, item in enumerate(get_files()):
        try:
            ua = f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{index}.0.0.0 Safari/537.36"
            session.headers["User-Agent"] = ua

            id = item
            data = load_file(item)
            title = data["title"]
            if "netflix" in data["tags"]:
                continue
            url = BASE_URL
            res, soup = get_data(url)
            if not res:
                continue
            hidden = soup.find(type="hidden")
            token = hidden["value"]

            url = URL.format(title=title, token=token)

            res, soup = get_data(url)

            print(f"INFO - Trying {id=}")

            try:
                content = soup.find(id="content")
            except Exception as exc:
                print(exc)

            if not content:
                print(f"INFO - Failed to find links for {id=}")
                continue

            for link in content.findAll("a"):
                # check the release year is correct
                if not f'({data["release_year"]})' in link.text:
                    continue

                url = link["href"]
                print(f"INFO - Found {url}")
                count += 1

                main_res, main_soup = get_data(url)

                country_list = main_soup.find(id="amp-list")
                if country_list is None:
                    print(f"ERROR - Couldn't find country list {id=}")
                    continue
                url = country_list["src"]
                res, soup = get_data(url)
                try:
                    countries = res.json()
                except Exception as exc:
                    print(exc, url, res.text)
                    continue

                if "nexflix" not in data:
                    data["netflix"] = []
                for country in countries["items"]:
                    try:
                        slug = country["url"].split("/")[-2]
                    except IndexError:
                        print(f"ERROR - Failed to get country {country=}")
                        continue
                    if slug not in data["netflix"]:
                        data["netflix"].append(slug)
                    print(f"INFO - Country for {id=} {country}")

                status = main_soup.find(id="streaming-status")
                if not status:
                    print(f"ERROR - Failed to get streaming info {id=}")
                    name = str(id) + ".html"
                    with open(FAILED / name, "w") as file:
                        file.write(main_res.text)
                    continue
                
                url = status["src"]
                res, soup = get_data(url)

                try:
                    netflix = res.json()
                except Exception as exc:
                    print(exc, url, res.text)
                    continue

                for item in netflix["items"]:
                    if item["title"] == "Watch on Netflix":
                        stream_url =  item["url"][0]
                        print(f"INFO - Added {stream_url}")
                        if stream_url not in data["links"]:
                            data["links"].append(stream_url)
                        data["tags"].append("netflix")
                break
            mov = METADATA / (str(id) + ".json")
            save_file(id, data, force=True)
        except Exception as exc:
            print(exc)

    print(count)

def netflix_just():
    session = requests.session()
    session.headers.update(just_headers)

    data = {}
    while just_payload["variables"]["offset"] < 5000:
        res = session.post(JUST_URL, json=just_payload)
        data = res.json()

        for edge in data["data"]["popularTitles"]["edges"]:
            imdb_id = edge["node"]["content"]["externalIds"]["imdbId"]
            netflix_url = edge["node"]["watchNowOffer"]["standardWebURL"]
            
            try:
                data = load_file(imdb_id)
            except KeyError:
                continue
            
            if "netflix" in data["tags"]:
                continue

            if netflix_url not in data["links"]:
                print(f"INFO - Got link for {imdb_id} {netflix_url}")
                data["links"].append(netflix_url)
            data["tags"].append("netflix")

            id = imdb_id

            mov = METADATA / (str(id) + ".json")
            save_file(id, data, force=True)

        just_payload["variables"]["offset"] += just_payload["variables"]["first"]
        time.sleep(5)

def get_cast(data):
    cast = []
    for key, value in data["props"]["pageProps"]["mappedData"].items():
        if "{" not in value:
            continue
        try:
            data = json.loads(value)
            if "castAndCrew" in data:
                for key, value in data["castAndCrew"].items():
                    if not value:
                        continue
                    for item in value:
                        cast.append(item["name"])
        except json.decoder.JSONDecodeError:
            pass

    return cast


def valid_hbo(movie: dict, data):
    """
    Extra HTTP call to ensure it's the movie we want
    """
    url = movie
    res, soup = get_data(url)
    
    if not soup:
        return False

    cast = get_cast(json.loads(soup.find(id="__NEXT_DATA__").text))
    try:
        if not data["release_year"] == soup.find("meta", {"property": "og:video:release_date"})["content"]:
            print(f"Wrong year for {data} at {movie}")
            return 
    except Exception as exc:
        print("ERROR - No release year", exc)
        return False

    pred = any(actor in cast for actor in data["actors"]) or any(staff in cast for staff in data["staff"])
    assertion = any(actor in cast for actor in data["actors"]) and any(staff in cast for staff in data["staff"])
    if not pred:
        print(f"WARNING - {movie=} is not {data['title']}")
        return False
    if not assertion:
        print(f"WARNING - Couldn't assert if this is the actual movie! {movie=}")
    return True

def hbo_max():
    BASE_URL = "https://www.max.com"
    url = "https://www.max.com/sitemap/movies"


    res, soup = get_data(url)
    links = soup.find(class_="my-5 container").findAll("a")
    counter = 0
    bad_counter = 0

    ids = set()

    for id in get_files():
        data = load_file(id)
        if "max" in data["tags"]:
            for link in data["links"].copy():
                if "max" in link and not valid_hbo(link, data):
                    mov = METADATA / Path(id + ".json")
                    data["links"].remove(link)
                    if "max" in data["tags"]:
                        data["tags"].remove("max")
                    bad_counter += 1
                    print(data, "is not valid hbo")
            continue
        for movie in links:
            link = BASE_URL + movie["href"]
            if data["title"] in movie.text and valid_hbo(link, data):
                print(f"INFO - Found link for {id=} {link=}")

                if "links" not in data:
                    data["links"] = []
                if link not in data["links"]:
                    data["links"].append(link)
                data["tags"].append("max")
                counter += 1
                ids.add(id)
                break
        mov = METADATA / (str(id) + ".json")
        save_file(id, data, force=True)
    print("hbo:", counter, bad_counter)

    # delete non-hbo movies
    #for id in get_files():
    #    mov = METADATA / Path(id + ".json")
    #    if id not in ids:
    #        mov.unlink()


def descriptions():
    BASE_URL = "https://imdb.com/title/{id}/plotsummary/"
    for id in get_files():
        data = load_file(id)
        if "description" in data:
            print(f"Skipping {id}")
            continue

        url = BASE_URL.format(id=id)
        res, soup = get_data(url, headers={"User-Agent": ""})
        try:
            desc = soup.find(**{"data-testid":"sub-section-synopsis"}).text
        except Exception as exc:
            print(exc)
            # no synopsis
            try:
                desc = soup.find(**{"data-testid":"sub-section-summaries"}).findAll("li")[1].next.next.next.next.next.next.text
            except Exception as exc:
                print(exc)
                continue
            # continue as usual if we got either synopsis or summary

        data["description"] = desc

        print(f"Description {id}: {desc}")

        save_file(id, data, force=True)

def run():
    #mov(MOVIES)
    #ratings(RATINGS)
    #actors()
    #fix_and_add_names()
    #covers()
    #descriptions()
    #amazon()
    #hbo_max()
    #netflix()
    netflix_just()

if __name__ == "__main__":
    run()
