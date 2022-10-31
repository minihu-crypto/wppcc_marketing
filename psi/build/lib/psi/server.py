import logging
import csv
from psi.pair import Address, make_pair
from psi import server_config
from psi.psi import Server


def start_server(address: str):                         #127.0.0.1:2345
    print('arrive start_server')
    peer_host, peer_port_str = address.split(":", 2)    #127.0.0.1, 2345
    peer_port = int(peer_port_str)

    local_address = Address(server_config.host, server_config.port)   #local  Address对象 127.0.0.1 1234
    peer_address = Address(peer_host, peer_port)                       #peer  Address对象 127.0.0.1 2345
    # with open(server_config.data, mode="r", encoding="utf-8") as f:
    #     data = [line.rstrip().encode("utf-8") for line in f] #对每一行去空格 并转换编码格式未 utf-8   打印data 看data啥样 csv->
    print(server_config.data)

    with open(server_config.data, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(reader)
        data = [line['id'].rstrip().encode("utf-8") for line in reader]

    with make_pair(local_address, peer_address) as pair:
        server = Server(pair, data)  # psi server
        logging.info("start prepare")
        server.prepare()  # prepare stage
        logging.info("finish prepare")

        logging.info("start intersection")
        res = server.intersect()  # intersect stage, res is the intersection
        logging.info("finish intersection")

        # save result
        res_strs = sorted([val.decode("utf-8") for val in res])
        # with open(server_config.result, mode="w", encoding="utf-8") as f:
        #     for line in res_strs:
        #         f.write(line)
        #         f.write("\n")
        with open(server_config.result, mode="w", encoding="utf-8") as f:
            for i in range(len(res_strs)):
                f.write(res_strs[i])
                if i < len(res_strs) - 1:
                    f.write(",")
        pair.barrier()
