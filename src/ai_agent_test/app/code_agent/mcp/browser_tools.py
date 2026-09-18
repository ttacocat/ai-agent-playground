import time
from mcp.server.fastmcp import FastMCP
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup, Comment
import re

mcp = FastMCP()

DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def slim_html(html: str) -> str:
    """瘦身 HTML：移除脚本、样式、图标、隐藏元素等冗余内容，减少 token 占用"""
    soup = BeautifulSoup(html, "html.parser")

    noisy_tags = [
        "script", "style", "link", "meta",
        "symbol", "path", "canvas", "svg",
        "noscript", "iframe", "footer", "header", "nav"
    ]
    for tag in soup(noisy_tags):
        tag.decompose()

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    for tag in soup.find_all(True):
        style = tag.get("style", "")
        if DISPLAY_NONE_RE.search(style):
            tag.extract()

    for tag in soup.find_all(True):
        if tag.name == "a":
            href = tag.attrs.get("href")
            if href is not None and ("javascript:" in href or href == "/"):
                tag.extract()
                continue
            elif href is not None:
                tag.attrs = {"href": href}
            else:
                tag.attrs = {}
        else:
            tag.attrs = {}

    html_str = str(soup)
    html_str = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())

    return html_str

def build_driver() -> webdriver.Chrome:
    service = Service(executable_path="/Users/wxy/bin/chromedriver")
    options = Options()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

    driver = webdriver.Chrome(service=service, options=options)
    print(f"成功连接到Chrome浏览器，当前URL：{driver.current_url}")
    return driver


def open_chrome():
    driver = build_driver()
    driver.get("https://www.bing.com")
    driver.execute_script("window.open('https://www.qq.com', '_blank');")

    # 获取所有句柄
    all_handles = driver.window_handles
    print(all_handles)

    # 切换句柄
    driver.switch_to.window(all_handles[0])


@mcp.tool(description="search query word in duckduckgo")
def search_in_duckduckgo(query: str) -> str | None:
    driver = build_driver()
    try:
        driver.set_window_size(1920, 1080)
        driver.get("https://html.duckduckgo.com/html/")

        search_box = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_box.send_keys(query)
        search_box.submit()

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.result"))
        )

        page_text_list = []
        max_pages = 3

        for i in range(max_pages):
            current_page = i + 1

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            results = driver.find_elements(By.CSS_SELECTOR, "div.result")

            page_html_list = []
            for r in results:
                raw_html = r.get_attribute("outerHTML")
                clean_html = slim_html(raw_html)
                page_html_list.append(clean_html)

            page_text_list.append("\n\n".join(page_html_list))
            print(f"第 {current_page} 页抓取完成，条目数: {len(results)}")

            if current_page == max_pages:
                break

            old_first_result = results[0] if results else None

            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, "input[value='Next'], .result--more__btn")
                    )
                )
            except Exception:
                print("没有更多页了，提前结束翻页")
                break

            next_button.click()

            if old_first_result is not None:
                WebDriverWait(driver, 10).until(EC.staleness_of(old_first_result))
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.result"))
            )

        return "\n\n---第X页分隔---\n\n".join(page_text_list)

    except Exception as e:
        print(f"运行出错: {e}")
        return None
    finally:
        driver.quit()


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # print(search_in_duckduckgo("张家界的天气"))
    # open_chrome()