from sshtunnel import SSHTunnelForwarder
import pymongo
import pyspark
def connection_create():
    MONGO_HOST = "10.4.41.45"
    MONGO_USER = "bdm"
    MONGO_PASS = "perezmartinez"
    MONGO_DB = "Landing_BDM_P2"

    server = SSHTunnelForwarder(
        MONGO_HOST,
        ssh_username=MONGO_USER,
        ssh_password=MONGO_PASS,
        remote_bind_address=('127.0.0.1', 27017)
    )
    server.start()
    client = pymongo.MongoClient('127.0.0.1', server.local_bind_port)
    mongo_db = client[MONGO_DB]
    return mongo_db, server

def drop_collection(collection):
    mongo_db, server = connection_create()
    mycol = mongo_db[collection]
    mycol.drop()

mongo_db, server = connection_create()
collections = mongo_db.list_collection_names()

db=mongo_db


for collection in collections:
    count = db[collection].count_documents({})
    print(f"Collection {collection} has {count} documents")

print(pyspark.__version__)


