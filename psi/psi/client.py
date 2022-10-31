from psi import client_config
from psi.pair import make_pair, Address
from psi.psi import Client
import logging
import csv


def start_client(address: str):
    print('arrive start_client')
    peer_host, peer_port_str = address.split(":", 2)
    peer_port = int(peer_port_str)

    local_address = Address(client_config.host, client_config.port)
    peer_address = Address(peer_host, peer_port)
    print("client_config:"+str(client_config))
    # with open(client_config.data, mode="r", encoding="utf-8") as f:
    #     data = [val.rstrip().encode("utf-8") for val in f]
    print(client_config.data)
    with open(client_config.data, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = [line['id'].rstrip().encode("utf-8") for line in reader]

    with make_pair(local_address, peer_address) as pair:
        client = Client(pair, data)

        logging.info("start prepare")
        client.prepare()  # prepare stage
        logging.info("finish prepare")

        logging.info("start intersection")
        res = client.intersect()  # intersect stage, res is the intersection
        logging.info("finish intersection")

        res_strs = sorted([val.decode("utf-8") for val in res])
        # with open(client_config.result, mode="w", encoding="utf-8") as f:
        #     for line in res_strs:
        #         f.write(line)
        #         f.write("\n")
        with open(client_config.result, mode="w", encoding="utf-8") as f:
            for i in range(len(res_strs)):
                f.write(res_strs[i])
                if i < len(res_strs) - 1:
                    f.write(",")
                

        pair.barrier()
