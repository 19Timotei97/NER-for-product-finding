# Step 1.
# Figured it would be best to check which websites I can actually use to crawl
import requests
import pandas as pd
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_csv', type=str, default='furniture stores pages.csv')
parser.add_argument('--path_for_text_file', type=str, default='good_links.txt')


class LinkChecker:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.links = []  # All links from csv file
        self.not_working_links = []  # not reachable at all
        self.good_links = []  # Sites which returned HTTP codes != 200 (OK)
        self.bad_links = []  # Sites which returned code 200

    def get_links_from_csv(self):
        df = pd.read_csv(self.csv_file)
        self.links = [df['max(page)'][i] for i in range(len(df))]

        # print(f"Got {len(self.links)} total links\n")
        logging.info(f"Got {len(self.links)} total links\n")

    def check_links(self):
        if len(self.links) == 0:
            # print("Make sure to run get_links_from_csv before running this method!")
            logging.error("Make sure to run get_links_from_csv before running this method!")

            return

        # Iterate over links
        # Yes, I know, a for-loop in Python, a crime. But I just wanted something simple.
        for link_ in self.links:
            try:
                resp = requests.head(link_)
            except Exception as e:  # I know, not the best way to catch exceptions, I should be more specific.
                # print(f"Got EXCEPTION {str(e)} for link {link_}!")
                logging.warning(f"Got EXCEPTION {str(e)} for link {link_}!")

                self.not_working_links.append(link_)
            else:
                # Add the link to the good list of links if it responded to the request
                if 200 == resp.status_code:
                    # print(f"Link {link_} is fine.")
                    logging.info(f"Link {link_} is fine.")

                    self.good_links.append(link_ + "\n")
                else:
                    # For any other code, I added them to a "bad" links list, even though some HTTP codes are not
                    #  necessarily bad.
                    print(f"Got CODE {resp.status_code} for link {link_}!")
                    logging.warning(f"Got CODE {resp.status_code} for link {link_}!")

                    self.bad_links.append(link_)

    def write_good_links_to_txt(self, path: str):
        if 0 == len(self.good_links):
            # print("No good links found!")
            logging.warning("No good links found!")
            return

        with open(path, 'w') as file:
            file.writelines(self.good_links)


def start_checking_links():
    args = parser.parse_args()
    # link_checker = LinkChecker('furniture stores pages.csv')
    link_checker = LinkChecker(args.path_to_csv)

    # Grab links from csv file
    link_checker.get_links_from_csv()

    # Check for broken / unreachable links
    link_checker.check_links()

    # print(f"Got {len(link_checker.not_working_links)} avoidable links!")
    # print(f"Got {len(link_checker.bad_links)} bad links (code != 200)!")
    # print(f"Got {len(link_checker.good_links)} good links!")

    logging.info(f"Got {len(link_checker.not_working_links)} avoidable links!")
    logging.info(f"Got {len(link_checker.bad_links)} bad links (code != 200)!")
    logging.info(f"Got {len(link_checker.good_links)} good links!")

    # link_checker.write_good_links_to_txt('good_links.txt')
    link_checker.write_good_links_to_txt(args.path_for_text_file)


if __name__ == '__main__':
    start_checking_links()
