"""PTT Web Scraper — extracted from PttStock_v2.ipynb.

Provides the PTTScraper class for scraping posts and comments
from PTT (批踢踢) bulletin board.
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor


class PTTScraper:
    base_url = "https://www.ptt.cc"

    def __init__(self, board):
        self.board = board
        self.url = self.base_url + f"/bbs/{board}/index.html"

    def get_post_content(self, post_url):
        """Fetch post content and push list from a post URL (relative path)."""
        soup = PTTScraper.get_soup(self.base_url + post_url)
        content = soup.find(id='main-content').text

        pushes = soup.find_all('div', class_='push')
        with ThreadPoolExecutor() as executor:
            push_list = list(executor.map(self.get_push, pushes))

        return content, push_list

    @staticmethod
    def get_push(push):
        """Extract tag/userid/content/datetime from a push element."""
        try:
            if push.find('span', class_='push-tag') is None:
                return dict()
            push_tag = push.find('span', class_='push-tag').text.strip()
            push_userid = push.find('span', class_='push-userid').text.strip()
            push_content = push.find('span', class_='push-content').text.strip().lstrip(":")
            push_ipdatetime = push.find('span', class_='push-ipdatetime').text.strip()
            push_dict = {
                "Tag": push_tag,
                "Userid": push_userid,
                "Content": push_content,
                "Ipdatetime": push_ipdatetime
            }
        except Exception as e:
            print(e)
            push_dict = dict()
        return push_dict

    @staticmethod
    def get_soup(url):
        """Fetch a URL and return a BeautifulSoup object."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            ),
        }
        cookies = {"over18": "1"}
        response = requests.get(url, headers=headers, cookies=cookies)
        return BeautifulSoup(response.text, 'html.parser')

    def fetch_post(self, url):
        """Fetch a full post: title, author, date, content, pushes."""
        soup = PTTScraper.get_soup(self.base_url + url)

        content = None
        author = None
        title = None
        date = None

        try:
            if soup.find(id='main-content') is not None:
                content = soup.find(id='main-content').text
                content = content.split('※ 發信站')[0]
            if soup.find(class_='article-meta-value') is not None:
                author = soup.find(class_='article-meta-value').text
                title = soup.find_all(class_='article-meta-value')[-2].text
                date_str = soup.find_all(class_='article-meta-value')[-1].text
                date = datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y')
        except Exception as e:
            print(self.base_url + url)
            print(e)

        pushes = soup.find_all('div', class_='push')
        with ThreadPoolExecutor() as executor:
            push_list = list(executor.map(self.get_push, pushes))

        return {
            'Title': title,
            'Author': author,
            'Date': date,
            'Content': content,
            'Link': url,
            'Pushes': push_list,
        }

    def get_data_current_page(self, soup=None, until_date=None, *args,
                              max_posts=1000, links_num=0):
        """Get post data from the current page, up to until_date.

        Parameters
        ----------
        soup : BeautifulSoup, optional
            Pre-fetched soup; if None, fetches current URL.
        until_date : datetime
            Stop collecting posts older than this date.
        *args : str
            Title keyword filters (1 or 2 keywords).
        max_posts : int
            Maximum number of posts to collect.
        links_num : int
            Number of links already collected (for pagination).

        Returns
        -------
        tuple : (data, reached_end, count)
        """
        if until_date is None:
            until_date = datetime.now()
        reach = False
        until_date = until_date.replace(hour=0, minute=0, second=0, microsecond=0)

        if soup is None:
            soup = PTTScraper.get_soup(self.url)

        links = []
        div_element = soup.find('div', {'class': 'r-list-sep'})

        if div_element is None:
            for entry in reversed(soup.select('.r-ent')):
                try:
                    title = entry.find("div", "title").text.strip()
                    if entry.find("div", "title").a is None:
                        continue
                    if len(args) == 2:
                        if not (args[0] in title and args[1] in title):
                            continue
                    elif len(args) == 1:
                        if args[0] not in title:
                            continue
                    date = entry.select('.date')[0].text.strip()
                    post_date = datetime.strptime(date, '%m/%d').replace(year=until_date.year)
                    if len(links) + links_num >= max_posts or post_date < until_date:
                        reach = True
                        break
                    links.append(entry.select('.title a')[0]['href'])
                except Exception as e:
                    print(e)
        else:
            previous_elements = [
                element for element in div_element.previous_siblings
                if element.name == 'div' and 'r-ent' in element.get('class', [])
            ]
            for element in reversed(previous_elements):
                title_link_element = element.find('a')
                if title_link_element:
                    title = title_link_element.text.strip()
                    if len(args) == 2:
                        if not (args[0] in title and args[1] in title):
                            continue
                    links.append(title_link_element.get('href'))
                date_element = element.find('div', {'class': 'date'})
                if date_element:
                    date = date_element.text.strip()
                post_date = datetime.strptime(date, '%m/%d').replace(year=until_date.year)
                if len(links) + links_num >= max_posts or post_date < until_date:
                    reach = True
                    break

        if 'post_date' not in locals():
            return [], False, 0

        print(post_date)

        with ThreadPoolExecutor() as executor:
            data = list(executor.map(self.fetch_post, links))

        return data, reach, len(links)

    def get_data_until(self, until_date, *args, max_posts=1000):
        """Get all posts published after until_date.

        Parameters
        ----------
        until_date : str or datetime
            Cutoff date (format '%m/%d' or datetime object).
        *args : str
            Title keyword filters.
        max_posts : int
            Maximum number of posts to collect.

        Returns
        -------
        list[dict] : List of post dictionaries.
        """
        data = []
        if not isinstance(until_date, datetime):
            date = datetime.strptime(until_date, '%m/%d').replace(year=datetime.now().year)
        else:
            date = until_date

        links_num = 0
        # Reset URL to current board index for fresh pagination
        self.url = self.base_url + f"/bbs/{self.board}/index.html"

        while True:
            soup = PTTScraper.get_soup(self.url)
            data_curr, date_end, num = self.get_data_current_page(
                soup, date, *args, max_posts=max_posts, links_num=links_num
            )
            data.extend(data_curr)
            if date_end:
                return data
            links_num += num

            prev_link = soup.find('a', string='‹ 上頁')['href']
            self.url = self.base_url + prev_link

        return data

    def get_data_days_before(self, delta_days, *args, max_posts=1000):
        """Get posts from delta_days days ago until now.

        Parameters
        ----------
        delta_days : int
            Number of days to look back.
        *args : str
            Title keyword filters.
        max_posts : int
            Maximum number of posts.

        Returns
        -------
        list[dict] : List of post dictionaries.
        """
        after_date = datetime.now() - timedelta(days=delta_days)
        return self.get_data_until(after_date, *args, max_posts=max_posts)

    def get_title_and_before_days(self, *args, delta_days, max_posts=1000):
        """Convenience wrapper for get_data_days_before with keyword args."""
        return self.get_data_days_before(delta_days, *args, max_posts=max_posts)
