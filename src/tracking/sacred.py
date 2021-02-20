import os
from sacred.observers import MongoObserver


def sacred_observer() -> MongoObserver:
    '''Constructs a Sacred observer from
    environment variables. '''
    user = os.environ['SACRED_USER']
    password = os.environ['SACRED_PASSWORD']
    host = os.environ['SACRED_HOST']
    database = os.environ['SACRED_DATABASE']
    url = f'mongodb+srv://{user}:{password}@{host}/{database}?retryWrites=true&w=majority'
    return MongoObserver(url, db_name=database)
