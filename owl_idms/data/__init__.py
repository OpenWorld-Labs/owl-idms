from . import cod_data

def get_loader(data_id, batch_size):
    if data_id == "cod_data":
        return cod_data.get_loader(batch_size)