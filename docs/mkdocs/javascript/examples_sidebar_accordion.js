(function () {
  function getPrimaryNav() {
    return document.querySelector('.md-nav--primary');
  }

  function getExamplesItem() {
    const primaryNav = getPrimaryNav();
    if (!primaryNav) {
      return null;
    }

    const topItems = primaryNav.querySelectorAll(':scope .md-nav__list > .md-nav__item--nested');
    for (const item of topItems) {
      const label = item.querySelector(':scope > label.md-nav__link, :scope > a.md-nav__link');
      if ((label?.textContent || '').trim() === 'Examples') {
        return item;
      }
    }
    return null;
  }

  function closeSiblingGroups(currentItem) {
    const siblings = currentItem.parentElement?.children || [];
    for (const sibling of siblings) {
      if (sibling === currentItem || !(sibling instanceof HTMLElement)) {
        continue;
      }
      const siblingToggle = sibling.querySelector(':scope > input.md-nav__toggle');
      if (siblingToggle instanceof HTMLInputElement) {
        siblingToggle.checked = false;
      }
    }
  }

  function handleToggleChange(event) {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }
    if (!target.classList.contains('md-nav__toggle') || !target.checked) {
      return;
    }

    const examplesItem = getExamplesItem();
    if (!examplesItem) {
      return;
    }

    const currentItem = target.closest('.md-nav__item--nested');
    if (!currentItem || !examplesItem.contains(currentItem) || currentItem === examplesItem) {
      return;
    }

    closeSiblingGroups(currentItem);
  }

  function applyExamplesAccordion() {
    const examplesItem = getExamplesItem();
    if (!examplesItem) {
      return;
    }

    const nav = getPrimaryNav();
    if (!nav || nav.dataset.examplesAccordionBound === '1') {
      return;
    }
    nav.dataset.examplesAccordionBound = '1';
    nav.addEventListener('change', handleToggleChange);
  }

  function init() {
    applyExamplesAccordion();
  }

  if (typeof window.document$ !== 'undefined' && typeof window.document$.subscribe === 'function') {
    window.document$.subscribe(init);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
