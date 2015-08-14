import urllib
import json
import random

zip_codes_file = open("zipcodes.dat", "r")
randomized_zip_codes = [(random.random(), line) for line in zip_codes_file.read().splitlines()]
randomized_zip_codes.sort()

LOCATIONS_TO_TRY = [zip_code for _, zip_code in randomized_zip_codes]
PETFINDER_API_KEY = "6147b7f8725e76cb84f2da7694e3aa5e"
URL_FORMAT = "http://api.petfinder.com/pet.find?key={key}&location={location}&count={count}&offset={offset}&animal=cat&format=json"
RECORDS_PER_REQUEST = 1000
IMAGES_TO_RETRIEVE = 2000000

def retrieve_dataset_from_petfinder():
    '''
        Gets all the images from a petfinder query, by looping through all
        specified locations, and every offset
    '''
    total_images_retrieved = 0
    for location in LOCATIONS_TO_TRY:
        if total_images_retrieved >= IMAGES_TO_RETRIEVE:
            break
        last_offset = 0
        keep_trying = True
        while keep_trying:
            print "Total images: ", total_images_retrieved
            url = URL_FORMAT.format(key=PETFINDER_API_KEY,
                                    location=location,
                                    count=RECORDS_PER_REQUEST,
                                    offset=last_offset)
            print "Getting images from: " + url
            response = urllib.urlopen(url)
            try:
                petfinderObject = json.loads(response.read())["petfinder"]
                if petfinderObject["header"]["status"]["code"]["$t"] != "100":
                    break
                last_offset = petfinderObject["lastOffset"]["$t"]
                petsList = petfinderObject["pets"]["pet"]
                for pet in petsList:
                    if "photos" in pet["media"]:
                        images = filter(lambda photo: photo["@size"] == "x",
                                      pet["media"]["photos"]["photo"])
                        for image in images:
                            urllib.urlretrieve(image["$t"], "data/" + pet["id"]["$t"] + "_" + image["@id"] + "_" + pet["sex"]["$t"] + ".jpg")
                            total_images_retrieved += 1
            except KeyError as ex:
                print "KeyError for " + url + ": " + str(ex)
                continue
            except ValueError as ex:
                print "ValueError for " + url + ": " + str(ex)
                continue
            except Exception as ex:
                print "Unknown error for " + url + ": " + str(ex)
                continue

retrieve_dataset_from_petfinder()
