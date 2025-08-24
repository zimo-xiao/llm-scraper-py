CLEANUP_JS = r"""
(() => {
  const elementsToRemove = [
    'script','style','noscript','iframe','svg','img','audio','video','canvas',
    'map','source','dialog','menu','menuitem','track','object','embed',
    'form','input','button','select','textarea','label','option','optgroup',
    'aside','footer','header','nav','head'
  ];
  const attributesToRemove = [
    'style','src','alt','title','role','aria-','tabindex','on','data-'
  ];
  const elementTree = document.querySelectorAll('*');
  elementTree.forEach((element) => {
    if (elementsToRemove.includes(element.tagName.toLowerCase())) {
      element.remove()
    }

    Array.from(element.attributes).forEach((attr) => {
      if (attributesToRemove.some((a) => attr.name.startsWith(a))) {
        element.removeAttribute(attr.name)
      }
    })
  });
})();
"""


TO_MARKDOWN_JS = r"""
(html) => {
  try {
    if (window.TurndownService) {
      const td = new TurndownService();
      return td.turndown(html);
    }
  } catch(e) {}
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  return tmp.innerText;
}
"""

TO_READABILITY_TEXT_JS = r"""
async () => {
  const mod = await import('https://cdn.skypack.dev/@mozilla/readability');
  const r = new mod.Readability(document).parse();
  return { title: r?.title || document.title, text: r?.textContent || '' };
}
"""

DEFAULT_PROMPT = (
    "You are a sophisticated web scraper. Extract the contents of the webpage."
)

DEFAULT_CODE_PROMPT = (
    "Provide a scraping function in JavaScript that extracts and returns data according to a schema "
    "from the current page. The function must be an IIFE. No comments or imports. No console.log. "
    "Output only runnable code."
)
