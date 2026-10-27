document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('detailColumns')?.addEventListener('change', e => document.body.classList.toggle('show-details', e.target.checked));
  const drawer = document.getElementById('flightDrawer');
  if (drawer) {
    let previousFocus;
    new MutationObserver(() => {
      const opened = drawer.classList.contains('active');
      drawer.setAttribute('aria-hidden', String(!opened));
      document.body.style.overflow = opened ? 'hidden' : '';
      if (opened) { previousFocus = document.activeElement; document.getElementById('btnCloseDrawer').focus(); }
      else if (previousFocus?.isConnected) previousFocus.focus();
    }).observe(drawer, {attributes:true,attributeFilter:['class']});
    drawer.addEventListener('keydown', e => {
      if(e.key !== 'Tab') return;
      const items = [...drawer.querySelectorAll('button,a[href],input,select,[tabindex="0"]')];
      const first = items[0], last = items[items.length-1];
      if(e.shiftKey && document.activeElement===first) {e.preventDefault();last.focus();}
      else if(!e.shiftKey && document.activeElement===last) {e.preventDefault();first.focus();}
    });
  }
  document.querySelectorAll('th[data-sort]').forEach(th => {
    th.tabIndex = 0; th.setAttribute('aria-sort','none');
    th.addEventListener('keydown',e => {if(e.key==='Enter'||e.key===' ') {e.preventDefault();th.click();}});

  });
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  function syncTabs() { tabs.forEach(tab => { const active=tab.classList.contains('active');tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1; }); }
  tabs.forEach((tab,i) => {
    const panelId = tab.getAttribute('onclick')?.match(/openTab\('([^']+)'/)?.[1];
    if(panelId) {tab.setAttribute('aria-controls',panelId);const panel=document.getElementById(panelId);panel.setAttribute('role','tabpanel');panel.setAttribute('aria-labelledby',tab.id);}
    tab.addEventListener('click',syncTabs);
    tab.addEventListener('keydown',e=>{ let next;if(e.key==='ArrowRight')next=(i+1)%tabs.length;if(e.key==='ArrowLeft')next=(i-1+tabs.length)%tabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=tabs.length-1;if(next!==undefined){e.preventDefault();tabs[next].click();tabs[next].focus();} });
  }); syncTabs();
  document.querySelectorAll('.control-group').forEach(group => {const control=group.querySelector('input,select');const text=group.querySelector('.control-label span');if(control&&text){text.id ||= control.id+'-label';control.setAttribute('aria-labelledby',text.id);} });
});
